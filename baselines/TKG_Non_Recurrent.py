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
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, g_list)
        triplets, labels = self.corrupter.sample_labels_val(g_list)
        return self.calc_metrics(per_graph_ent_embeds, t_list, triplets, labels)

    def forward(self, t_list, reverse=False):
        kld_loss = 0
        reconstruct_loss = 0
        g_list = [self.graph_dict_train[i.item()] for i in t_list]

        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, g_list)

        triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.samples_labels_train(t_list, g_list)

        for i, ent_embed in enumerate(per_graph_ent_embeds):
            loss_tail = self.train_link_prediction(ent_embed, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        return reconstruct_loss, kld_loss
