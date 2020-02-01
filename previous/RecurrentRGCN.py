from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import move_dgl_to_cuda, comp_deg_norm
from utils.scores import *
from previous.TKG_Recurrent_Module import TKG_Recurrent_Module
from torch import nn


class RecurrentRGCN(TKG_Recurrent_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(RecurrentRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.rnn = nn.GRU(input_size=self.embed_size * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.args.dropout)
        self.h0 = nn.Parameter(torch.Tensor(self.num_layers, 1, self.hidden_size))

    def build_model(self):
        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)

    def get_per_graph_ent_embeds(self, g_batched_list_t, cur_h, node_sizes, val=False):
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
                norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(norm).view(-1, 1)})
                sg.edata['type_s'] = rel[np.concatenate((graph_split_ids, graph_split_rev_ids))]
                sg.ids = g.ids
                sampled_graph_list.append(sg)

        batched_graph = dgl.batch(sampled_graph_list)
        expanded_h = torch.cat(
                [cur_h[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)

        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)

        batched_graph.ndata['h'] = torch.cat([ent_embeds, expanded_h], dim=-1)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        enc_ent_mean_graph = self.ent_encoder(batched_graph, reverse=False)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)
        return per_graph_ent_embeds