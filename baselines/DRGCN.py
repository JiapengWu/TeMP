from baselines.StaticRGCN import StaticRGCN
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import comp_deg_norm, move_dgl_to_cuda
from utils.scores import *
import torch.nn as nn
import torch

class DRGCN(StaticRGCN):

    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(DRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len
        self.num_pos_facts = self.args.num_pos_facts
        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)
        self.w_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.hidden_size))
        self.b_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.hidden_size))
        nn.init.xavier_uniform_(self.w_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.b_ent_embeds, gain=nn.init.calculate_gain('relu'))


    def get_per_graph_ent_embeds(self, t_list, graph_train_list, val=False):
        if val:
            sampled_graph_list = graph_train_list
        else:
            sampled_graph_list = []
            for g in graph_train_list:
                src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
                half_num_nodes = int(src.shape[0] / 2)
                graph_split_ids = np.random.choice(np.arange(half_num_nodes),
                                                   size=int(0.5 * half_num_nodes), replace=False)
                graph_split_rev_ids = graph_split_ids + half_num_nodes

                sg = g.edge_subgraph(np.concatenate((graph_split_ids, graph_split_rev_ids)), preserve_nodes=True)
                norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(norm).view(-1, 1)})
                sg.edata['type_s'] = rel[np.concatenate((graph_split_ids, graph_split_rev_ids))]
                sg.ids = g.ids
                sampled_graph_list.append(sg)
        time_embeds = []
        for t, g in zip(t_list, graph_train_list):
            temp_ent_embeds = torch.sin(t * self.w_ent_embeds[g.ndata['id']].view(-1, self.embed_size) +
                          self.b_ent_embeds[g.ndata['id']].view(-1, self.embed_size))
            time_embeds.append(temp_ent_embeds)
        batched_graph = dgl.batch(sampled_graph_list)

        time_embeds = torch.cat(time_embeds, dim=0)
        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        batched_graph.ndata['h'] = torch.cat([ent_embeds, time_embeds], dim=-1)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        node_sizes = [len(g.nodes()) for g in graph_train_list]
        enc_ent_mean_graph = self.ent_encoder(batched_graph, reverse=False)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)
        return per_graph_ent_embeds
