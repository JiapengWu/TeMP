from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import torch
import dgl
import numpy as np
from utils.scores import *
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent
import math
import time
import pdb


class AtiSE(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(AtiSE, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.static_embed_size = math.floor(0.5 * self.embed_size)
        self.temporal_embed_size = self.embed_size - self.static_embed_size
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.w_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))
        self.b_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.b_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))

    def get_per_graph_ent_embeds(self, t_list, g_list):
        per_graph_ent_embeds = []
        for t, g in zip(t_list, g_list):
            static_ent_embeds = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
            ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)
            temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size) +
                                    self.b_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size))
            ent_embeds = static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=1)
            per_graph_ent_embeds.append(ent_embeds)
        return per_graph_ent_embeds
