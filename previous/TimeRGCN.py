from torch import nn
from models.TKG_Module import TKG_Module
from utils.utils import filter_none
import torch


class TimeRGCN(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TimeRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.num_layers = self.args.num_layers
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len

        self.h0 = nn.Parameter(torch.Tensor(self.num_layers, 1, self.hidden_size))
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

        self.rnn = nn.GRU(input_size=self.embed_size * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.args.dropout)

