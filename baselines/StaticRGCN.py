from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import reparametrize, move_dgl_to_cuda
from utils.scores import *
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent


class StaticRGCN(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times):
        super(StaticRGCN, self).__init__(args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times)

    def build_model(self):
        self.half_size = int(self.embed_size / 2)
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len
        self.num_pos_facts = self.args.num_pos_facts

        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels, static=True)
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def get_per_graph_ent_embeds(self, t_list, g_list):
        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        node_sizes = [len(g.nodes()) for g in g_list]
        enc_ent_mean_graph = self.ent_encoder(batched_graph, reverse=False)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)
        return per_graph_ent_embeds

