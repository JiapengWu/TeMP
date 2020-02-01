from torch import nn
from utils.scores import *
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent
import math


class DiachronicEmbedding(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(DiachronicEmbedding, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.static_embed_size = math.floor(0.5 * self.embed_size)
        self.temporal_embed_size = self.embed_size - self.static_embed_size

        self.w_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))
        self.b_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))

        nn.init.xavier_uniform_(self.w_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.b_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))

    def get_all_embeds_Gt(self, t):
        static_ent_embeds = self.ent_embeds
        ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)

        temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds.view(-1, self.temporal_embed_size) +
                                    self.b_temp_ent_embeds.view(-1, self.temporal_embed_size))
        return static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=-1)

    def get_per_graph_ent_embeds(self, t, g):
        static_ent_embeds = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
        ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)
        temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size) +
                                self.b_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size))

        return static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=-1)
