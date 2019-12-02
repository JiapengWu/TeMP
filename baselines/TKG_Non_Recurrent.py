import numpy as np
import torch.nn.functional as F
from models.TKG_Module import TKG_Module
import time
from utils.evaluation import calc_metrics
import pdb


class TKG_Non_Recurrent(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times):
        super(TKG_Non_Recurrent, self).__init__(args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times)

    def evaluate(self, t_list, reverse=False):
        g_list = [self.graph_dict_total[i.item()] for i in t_list]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, g_list)
        triplets, labels = self.corrupter.sample_labels_val(g_list)
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        for i, ent_embed in enumerate(per_graph_ent_embeds):
            mrr, hit_1, hit_3, hit_10 = calc_metrics(ent_embed, self.rel_embeds, triplets[i])
            loss = self.link_classification_loss(ent_embed, triplets[i], labels[i])
            mrrs.append(mrr)
            hit_1s.append(hit_1)
            hit_3s.append(hit_3)
            hit_10s.append(hit_10)
            losses.append(loss.item())
        return np.mean(mrrs), np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_10s), np.sum(losses), 0

    def forward(self, t_list, reverse=False):
        kld_loss = 0
        reconstruct_loss = 0
        g_list = [self.graph_dict_total[i.item()] for i in t_list]

        per_graph_ent_embeds = self.get_per_graph_ent_embeds(t_list, g_list)
        start = time.time()
        triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.samples_labels_train(t_list, g_list)

        for i, ent_embed in enumerate(per_graph_ent_embeds):
            loss_tail = self.train_link_prediction(ent_embed, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        # print("Graph reconstruction: {}".format(time.time() - start))
        return reconstruct_loss, kld_loss
