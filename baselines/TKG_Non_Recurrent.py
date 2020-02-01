from models.TKG_Module import TKG_Module
import time
import torch
import torch.nn as nn


class TKG_Non_Recurrent(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Non_Recurrent, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_list = [graph_dict[i.item()] for i in t_list]
        return self.calc_metrics(g_list, t_list)

    def forward(self, t_list):
        reconstruct_loss = 0
        g_list = [self.graph_dict_train[i.item()] for i in t_list]

        for t, g in zip(t_list, g_list):
            ent_embed = self.get_per_graph_ent_embeds(t, g)
            all_embeds_g = self.get_all_embeds_Gt(t)
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        return reconstruct_loss
