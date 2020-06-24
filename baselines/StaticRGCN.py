from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import comp_deg_norm, move_dgl_to_cuda
from utils.scores import *
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent
from utils.utils import cuda, node_norm_to_edge_norm


class StaticRGCN(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(StaticRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.train_seq_len
        self.num_pos_facts = self.args.num_pos_facts
        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels, self.total_time)

    def evaluate(self, t_list, val=True):
        per_graph_ent_embeds, g_list = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds, g_list, t_list)

    def evaluate_embed(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        graph_train_list = [self.graph_dict_train[i.item()] for i in t_list]
        g_list = [graph_dict[i.item()] for i in t_list]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, graph_train_list, val=True)
        return per_graph_ent_embeds, g_list

    def train_embed(self, t_list):
        graph_train_list = [self.graph_dict_train[i.item()] for i in t_list]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, graph_train_list, val=True)
        return per_graph_ent_embeds, graph_train_list

    def forward(self, t_list):
        reconstruct_loss = 0
        g_list = [self.graph_dict_train[i.item()] for i in t_list]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, g_list)
        for t, g, ent_embed in zip(t_list, g_list, per_graph_ent_embeds):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            all_embeds_g = self.get_all_embeds_Gt(t, g, ent_embed)
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        return reconstruct_loss

    def get_all_embeds_Gt(self, t, g, convoluted_embeds):
        all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)

        if self.args.use_embed_for_non_active:
            all_embeds_g[:] = self.ent_embeds[:]
        else:
            all_embeds_g[:] = self.ent_encoder.forward_isolated(self.ent_embeds, t)[:]

        for k, v in g.ids.items():
            all_embeds_g[v] = convoluted_embeds[k]
        return all_embeds_g

    def get_per_graph_ent_embeds(self, t_list, graph_train_list, val=False):
        if val:
            sampled_graph_list = graph_train_list
        else:
            # TODO: modify half_num_nodes
            sampled_graph_list = []
            for g in graph_train_list:
                src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
                half_num_nodes = int(src.shape[0] / 2)
                # graph_split_ids = np.random.choice(np.arange(half_num_nodes),
                #                                    size=int(0.5 * half_num_nodes), replace=False)
                # graph_split_rev_ids = graph_split_ids + half_num_nodes
                # sg = g.edge_subgraph(np.concatenate((graph_split_ids, graph_split_rev_ids)), preserve_nodes=True)
                total_idx = np.random.choice(np.arange(src.shape[0]), size=int(0.5 * src.shape[0]), replace=False)
                sg = g.edge_subgraph(total_idx, preserve_nodes=True)
                node_norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
                sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
                sg.edata['type_s'] = rel[total_idx]
                sg.ids = g.ids
                sampled_graph_list.append(sg)
        batched_graph = dgl.batch(sampled_graph_list)
        batched_graph.ndata['h'] = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        node_sizes = [len(g.nodes()) for g in graph_train_list]
        enc_ent_mean_graph = self.ent_encoder(batched_graph, t_list, node_sizes)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)
        return per_graph_ent_embeds

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):
            all_embeds_g = self.get_all_embeds_Gt(t, g, ent_embed)
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

        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)

