from models.DynamicRGCN import DynamicRGCN
import dgl
import numpy as np
from utils.utils import comp_deg_norm, move_dgl_to_cuda
import torch.nn as nn
import torch
from utils.utils import node_norm_to_edge_norm
import math


class TimeDynamicRGCN(DynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TimeDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        super().build_model()
        self.static_embed_size = math.floor(0.8 * self.embed_size)
        self.temporal_embed_size = self.embed_size - self.static_embed_size

        self.w_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))
        self.b_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))

        nn.init.xavier_uniform_(self.w_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.b_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))

    def get_all_embeds_Gt(self, convoluted_embeds, g, t, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor):
        all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)

        static_ent_embeds = self.ent_embeds
        ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)

        temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds.view(-1, self.temporal_embed_size) +
                                    self.b_temp_ent_embeds.view(-1, self.temporal_embed_size))
        input_embeddings = static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=-1)
        if self.args.use_embed_for_non_active:
            all_embeds_g[:] = input_embeddings[:]
        else:
            all_embeds_g = self.ent_encoder.forward_isolated(input_embeddings, first_prev_graph_embeds,
                                                             second_prev_graph_embeds, time_diff_tensor.unsqueeze(-1))

        for k, v in g.ids.items():
            all_embeds_g[v] = convoluted_embeds[k]
        return all_embeds_g

    def get_per_graph_ent_embeds(self, g_batched_list_t, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, val=False):
        if val:
            sampled_graph_list = g_batched_list_t
        else:
            sampled_graph_list = []
            for g in g_batched_list_t:
                src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
                half_num_nodes = int(src.shape[0] / 2)
                # graph_split_ids = np.random.choice(np.arange(half_num_nodes), size=int(0.5 * src.shape[0]), replace=False)
                # graph_split_rev_ids = graph_split_ids + half_num_nodes
                # total_idx = np.concatenate((graph_split_ids, graph_split_rev_ids))
                total_idx = np.random.choice(np.arange(src.shape[0]), size=int(0.5 * src.shape[0]), replace=False)
                sg = g.edge_subgraph(total_idx, preserve_nodes=True)
                node_norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
                sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
                sg.edata['type_s'] = rel[total_idx]
                sg.ids = g.ids
                sampled_graph_list.append(sg)

        ent_embeds = []
        for t, g in zip(time_batched_list_t, g_batched_list_t):
            static_ent_embeds = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
            ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)
            temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size) +
                                        self.b_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size))

            ent_embeds.append(static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=-1))

        batched_graph = dgl.batch(sampled_graph_list)
        batched_graph.ndata['h'] = torch.cat(ent_embeds, dim=0)

        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_graph, second_layer_graph = self.ent_encoder(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor)

        first_layer_embeds = first_layer_graph.ndata['h']
        second_layer_embeds = second_layer_graph.ndata['h']
        return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)