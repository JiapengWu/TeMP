from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import torch
import dgl
import numpy as np
from utils.scores import *
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent


class Static(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times):
        super(Static, self).__init__(args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times)

    def build_model(self):
        self.negative_rate = self.args.negative_rate
        self.calc_score = {'distmult': distmult, 'complex': complex}[self.args.score_function]
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def get_per_graph_ent_embeds(self, t_list, g_list):
        batched_graph = dgl.batch(g_list)
        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        node_sizes = [len(g.nodes()) for g in g_list]
        return ent_embeds.split(node_sizes)