from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import torch
import dgl
import numpy as np
from utils.scores import *
from utils.evaluation import calc_metrics
from models.TKG_Module import TKG_Module

class Static(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times):
        super(Static, self).__init__(args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times)

    def build_model(self):
        self.negative_rate = self.args.negative_rate
        self.calc_score = {'distmult': distmult, 'complex': complex}[self.args.score_function]
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def link_classification_loss(self, ent_embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = self.rel_embeds[triplets[:, 1]]
        o = ent_embed[triplets[:, 2]]
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def get_per_graph_ent_embeds(self, g_list):
        batched_graph = dgl.batch(g_list)
        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        node_sizes = [len(g.nodes()) for g in g_list]
        return ent_embeds.split(node_sizes)

    def evaluate(self, t_list, reverse=False):
        g_list = [self.graph_dict_total[i.item()] for i in t_list]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_list)
        triplets, labels = self.corrupter.samples_labels(t_list, g_list, val=True)
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
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_list)
        triplets, labels = self.corrupter.samples_labels(t_list, g_list)
        # run distmult decoding
        for i, ent_embed in enumerate(per_graph_ent_embeds):
            loss = self.link_classification_loss(ent_embed, triplets[i], labels[i])
            reconstruct_loss += loss
        return reconstruct_loss, kld_loss
