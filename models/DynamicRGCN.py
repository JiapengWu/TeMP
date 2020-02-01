from torch import nn
from models.TKG_Module import TKG_Module
from utils.utils import filter_none
import torch
from models.DRGCN import DRGCN
from models.RRGCN import RRGCN
import numpy as np
from utils.utils import move_dgl_to_cuda, comp_deg_norm, node_norm_to_edge_norm, cuda
from utils.scores import *
import dgl

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import TimeDataset


class DynamicRGCN(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(DynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.num_layers = self.args.num_layers
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len

        # self.train_seq_len = len(self.total_time)
        # self.test_seq_len = len(self.total_time)

        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def _dataloader(self, times, mode="train"):
        # when using multi-node (ddp) we need to add the  datasampler
        dataset = TimeDataset(times)
        batch_size = self.args.batch_size
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        if mode == 'train':
            should_shuffle = train_sampler is None
        else:
            should_shuffle = False

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size if mode == 'train' else 10000,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        return self._dataloader(self.total_time)

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(self.total_time, 'val')

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.total_time, 'val')

    def build_model(self):
        if self.args.module == "RRGCN":
            self.ent_encoder = RRGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)
        else:
            self.ent_encoder = DRGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)

    def get_prev_embeddings(self, g_batched_list_t, history_embeddings, start_time_tensor, t):
        first_layer_prev_embeddings = []
        second_layer_prev_embeddings = []
        time_diff_tensor = []
        for i, graph in enumerate(g_batched_list_t):
            node_idx = graph.ndata['id']
            first_layer_prev_embeddings.append(history_embeddings[i][0][node_idx].view(-1, self.embed_size))
            second_layer_prev_embeddings.append(history_embeddings[i][1][node_idx].view(-1, self.embed_size))
            time_diff_tensor.append(t - start_time_tensor[i][node_idx])
        # if t >= 1: pdb.set_trace()

        return torch.cat(first_layer_prev_embeddings), torch.cat(second_layer_prev_embeddings), torch.cat(time_diff_tensor)

    def update_time_diff_hist_embeddings(self, first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, time, bsz):
        # import pdb; pdb.set_trace()
        res = start_time_tensor.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        for i in range(len(first_per_graph_ent_embeds)):
            idx = g_batched_list_t[i].ndata['id'].squeeze()
            res[i][0][idx] = first_per_graph_ent_embeds[i]
            res[i][1][idx] = second_per_graph_ent_embeds[i]
            start_time_tensor[i][idx] = time
        return res

    def get_all_embeds_Gt(self, g, convoluted_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor):
        all_embeds_g = self.ent_encoder.forward_isolated(self.ent_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor.unsqueeze(-1))
        for k, v in g.ids.items():
            all_embeds_g[v] = convoluted_embeds[k]
        return all_embeds_g

    def get_per_graph_ent_embeds(self, g_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, val=False):
        if val:
            sampled_graph_list = g_batched_list_t
        else:
            sampled_graph_list = []
            for g in g_batched_list_t:
                src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
                half_num_nodes = int(src.shape[0] / 2)
                graph_split_ids = np.random.choice(np.arange(half_num_nodes),
                                                   size=int(0.5 * half_num_nodes), replace=False)
                graph_split_rev_ids = graph_split_ids + half_num_nodes

                sg = g.edge_subgraph(np.concatenate((graph_split_ids, graph_split_rev_ids)), preserve_nodes=True)
                node_norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
                sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
                sg.edata['type_s'] = rel[np.concatenate((graph_split_ids, graph_split_rev_ids))]
                sg.ids = g.ids
                sampled_graph_list.append(sg)

        batched_graph = dgl.batch(sampled_graph_list)
        batched_graph.ndata['h'] = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)

        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_graph, second_layer_graph = self.ent_encoder(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor)

        first_layer_embeds = first_layer_graph.ndata['h']
        second_layer_embeds = second_layer_graph.ndata['h']
        return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_val_vars(self, g_batched_list, t):
        g_batched_list_t = filter_none(g_batched_list[t])
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in g_batched_list_t]
        return g_batched_list_t, node_sizes

    def evaluate(self, t_list, val=True):
        # print(t_list)
        t_list = torch.tensor([len(self.total_time) - 1])
        self.test_seq_len = len(self.total_time)
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_batched_list, val_time_list = self.get_batch_graph_list(t_list, self.test_seq_len, graph_dict)
        # import pdb; pdb.set_trace()
        bsz = len(g_batched_list[-1])
        hist_embeddings = self.ent_embeds.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        start_time_tensor = self.ent_embeds.new_zeros(bsz, self.num_ents)
        all_ranks = []
        all_losses = []
        for t in range(self.test_seq_len):
            print(t, end='\r')
            g_train_batched_list_t, node_sizes = self.get_val_vars(g_train_batched_list, t)
            g_val_batched_list_t, _ = self.get_val_vars(g_batched_list, t)
            # if len(g_train_batched_list_t) == 0: continue
            first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(g_train_batched_list_t, hist_embeddings, start_time_tensor, t)
            first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_train_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, val=True)

            ranks, losses = self.calc_metrics(second_per_graph_ent_embeds, g_val_batched_list_t, time_list[t], hist_embeddings, start_time_tensor, t)
            hist_embeddings = self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_train_batched_list_t, t, bsz)
            # import pdb; pdb.set_trace()
            all_ranks.append(ranks)
            all_losses.append(losses)
        return torch.cat(all_ranks), np.mean(all_losses)

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        # import pdb; pdb.set_trace()
        bsz = len(g_batched_list[-1])
        hist_embeddings = self.ent_embeds.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        start_time_tensor = self.ent_embeds.new_zeros(bsz, self.num_ents)
        # print(time_batched_list)

        # # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        for t in range(self.train_seq_len):
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, t)
            time_batched_list_t = filter_none(time_batched_list[t])

            if len(g_batched_list_t) == 0: continue
            first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(g_batched_list_t, hist_embeddings, start_time_tensor, t)
            first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, val=True)
            # import pdb; pdb.set_trace()
            ###
            i = 0
            all_embeds_hist = []
            for tim, g, ent_embed in zip(time_batched_list_t, g_batched_list_t, second_per_graph_ent_embeds):
                triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(tim, g, self.num_ents)
                all_embeds_g = self.get_all_embeds_Gt(g, ent_embed, hist_embeddings[i][0], hist_embeddings[i][1], t - start_time_tensor[i])
                all_embeds_hist.append(all_embeds_g)
                loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
                loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
                reconstruct_loss += loss_tail + loss_head
                i += 1

            if t < self.train_seq_len - 1:
                hist_embeddings = self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, t, bsz)
        return reconstruct_loss

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list, hist_embeddings, start_time_tensor, time):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):

            all_embeds_g = self.get_all_embeds_Gt(g, ent_embed, hist_embeddings[i][0], hist_embeddings[i][1], time - start_time_tensor[i])

            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                label = cuda(label)
            if index_sample.shape[0] == 0: continue
            rank = self.evaluater.calc_metrics_single_graph(ent_embed, self.rel_embeds, all_embeds_g, index_sample, g, t)
            loss = self.link_classification_loss(ent_embed, self.rel_embeds, index_sample, label)
            ranks.append(rank)
            losses.append(loss.item())
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)

