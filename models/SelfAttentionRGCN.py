from torch import nn
import numpy as np
from utils.utils import move_dgl_to_cuda, cuda, filter_none
from utils.scores import *

from models.DynamicRGCN import DynamicRGCN
from models.SARGCN import SARGCN
import pdb
import torch
from utils.evaluation import EvaluationFilter


class SelfAttentionRGCN(DynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=EvaluationFilter):
        super(SelfAttentionRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)
        self.EMA = self.args.EMA
        if self.EMA:
            self.alpha = nn.Parameter(torch.Tensor(self.embed_size, 1))

    def build_model(self):
        self.ent_encoder = SARGCN(self.args, self.hidden_size, self.embed_size, self.num_rels, self.total_time)
        self.time_diff_test = torch.tensor(list(range(self.test_seq_len - 1, -1, -1))).float()
        self.time_diff_train = torch.tensor(list(range(self.train_seq_len - 1, -1, -1))).float()
        if self.use_cuda:
            self.time_diff_test = cuda(self.time_diff_test)
            self.time_diff_train = cuda(self.time_diff_train)

    def get_all_embeds_Gt(self, convoluted_embeds, g, t, first_prev_graph_embeds, second_prev_graph_embeds, attn_mask, val=False):
        # input_embeddings = self.ent_embeds + self.time_embed[t]
        all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)
        if self.args.use_embed_for_non_active:
            all_embeds_g[:] = self.ent_embeds[:]
        else:
            if self.EMA:
                all_embeds_g = self.ent_encoder.forward_ema_isolated(self.ent_embeds, second_prev_graph_embeds.transpose(0, 1), t, torch.sigmoid(self.alpha), self.train_seq_len)
            else:
                all_embeds_g = self.ent_encoder.forward_isolated(self.ent_embeds, first_prev_graph_embeds.transpose(0, 1), second_prev_graph_embeds.transpose(0, 1),
                                                                 self.time_diff_test if val else self.time_diff_train, attn_mask.transpose(0, 1), t)

        keys = np.array(list(g.ids.keys()))
        values = np.array(list(g.ids.values()))

        all_embeds_g[values] = convoluted_embeds[keys]
        return all_embeds_g

    def get_per_graph_ent_dropout_embeds(self, cur_time_list, target_time_list, node_sizes):
        batched_graph = self.get_batch_graph_dropout_embeds(filter_none(cur_time_list), target_time_list)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_embeds, second_layer_embeds = self.ent_encoder(batched_graph, cur_time_list, node_sizes)

        return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_per_graph_ent_embeds(self, g_batched_list_t, time_batched_list_t, node_sizes, full=False, rate=0.5):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_embeds, second_layer_embeds = self.ent_encoder(batched_graph, time_batched_list_t, node_sizes)

        # first_layer_embeds = first_layer_graph.ndata['h']
        # second_layer_embeds = second_layer_graph.ndata['h']
        '''
        first_layer_embeds_split = []
        second_layer_embeds_split = []

        for t, first_layer_embed, second_layer_embed in zip(time_batched_list_t, first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)):
            first_layer_embeds_split.append(first_layer_embed + self.time_embed[t])
            second_layer_embeds_split.append(second_layer_embed + self.time_embed[t])
        return first_layer_embeds_split, second_layer_embeds_split
        '''
        return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_prev_embeddings(self, g_batched_list_t, hist_embeddings, attn_mask):
        first_layer_prev_embeddings = []
        second_layer_prev_embeddings = []
        local_attn_mask = []
        for i, graph in enumerate(g_batched_list_t):
            node_idx = graph.ndata['id'].squeeze()
            # pdb.set_trace()
            first_layer_prev_embeddings.append(hist_embeddings[:, i, 0, node_idx])
            second_layer_prev_embeddings.append(hist_embeddings[:, i, 1, node_idx])
            local_attn_mask.append(attn_mask[:, i, node_idx])

        return torch.cat(first_layer_prev_embeddings, dim=1).transpose(0, 1), torch.cat(second_layer_prev_embeddings, dim=1).transpose(0, 1), torch.cat(local_attn_mask, dim=1).transpose(0, 1)

    def get_final_graph_embeds(self, g_batched_list_t, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full, rate=0.5, val=False):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_prev_embeddings, second_layer_prev_embeddings, local_attn_mask = self.get_prev_embeddings(g_batched_list_t, hist_embeddings, attn_mask)
        # if self.EMA:
        #     second_layer_embeds = self.ent_encoder.forward_ema(batched_graph, second_layer_prev_embeddings, time_batched_list_t, node_sizes, torch.sigmoid(self.alpha), self.train_seq_len)
        second_layer_embeds = self.ent_encoder.forward_final(batched_graph, first_layer_prev_embeddings, second_layer_prev_embeddings,
                                                             self.time_diff_test if val else self.time_diff_train, local_attn_mask, time_batched_list_t, node_sizes)
        return second_layer_embeds.split(node_sizes)

    def update_time_diff_hist_embeddings(self, first_per_graph_ent_embeds, second_per_graph_ent_embeds, hist_embeddings, g_batched_list_t, cur_t, attn_mask, bsz):
        for i in range(len(first_per_graph_ent_embeds)):
            idx = g_batched_list_t[i].ndata['id'].squeeze()
            attn_mask[cur_t][i][idx] = 0
            hist_embeddings[cur_t][i][0][idx] = first_per_graph_ent_embeds[i]
            hist_embeddings[cur_t][i][1][idx] = second_per_graph_ent_embeds[i]

    def pre_forward(self, g_batched_list, time_batched_list, val=False):
        seq_len = self.test_seq_len if val else self.train_seq_len
        bsz = len(g_batched_list[0])
        target_time_batched_list = time_batched_list[-1]
        hist_embeddings = self.ent_embeds.new_zeros(seq_len - 1, bsz, 2, self.num_ents, self.embed_size)
        attn_mask = self.ent_embeds.new_zeros(seq_len, bsz, self.num_ents) - 10e9
        attn_mask[-1] = 0
        full = val or not self.args.random_dropout
        for cur_t in range(seq_len - 1):
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, cur_t)
            if len(g_batched_list_t) == 0: continue
            if self.edge_dropout and not val:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_dropout_embeds(time_batched_list[cur_t], target_time_batched_list, node_sizes)
            else:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, time_batched_list[cur_t], node_sizes, full=full, rate=0.8)
            self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, hist_embeddings, g_batched_list_t, cur_t, attn_mask, bsz)
        return hist_embeddings, attn_mask

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings, attn_mask = self.pre_forward(g_batched_list, time_batched_list, val=False)

        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full=False, val=False)

        i = 0
        for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings[:, i, 0], hist_embeddings[:, i, 1], attn_mask[:, i], val=False)
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, _ = self.get_batch_graph_list(t_list, 1, graph_dict)
        hist_embeddings, attn_mask = self.pre_forward(g_train_batched_list, time_list, val=True)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs = g_train_batched_list[-1]

        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_list[-1], node_sizes, hist_embeddings, attn_mask, full=True, val=True)
        return self.calc_metrics(per_graph_ent_embeds, test_graphs, time_list[-1], hist_embeddings, attn_mask)

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list, hist_embeddings, attn_mask):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings[:, i, 0], hist_embeddings[:, i, 1], attn_mask[:, i], val=True)
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
