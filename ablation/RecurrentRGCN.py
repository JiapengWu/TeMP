from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import reparametrize, move_dgl_to_cuda
from utils.scores import *
from ablation.TKG_Recurrent_Module import TKG_Recurrent_Module


class RecurrentRGCN(TKG_Recurrent_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(RecurrentRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)

    def get_per_graph_ent_embeds(self, g_batched_list_t, cur_h, node_sizes):
        batched_graph = dgl.batch(g_batched_list_t)
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