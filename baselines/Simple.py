import dgl
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent
from utils.scores import simple, distmult
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from utils.CorrptTriples import CorruptTriples
from utils.evaluation import EvaluationFilter


class SimpleEvaluationFilter(EvaluationFilter):

    def calc_metrics_single_graph(self, ent_mean, ent_mean_inv, rel_enc_means, test_triplets, time, eval_bz=100):
        with torch.no_grad():
            s = test_triplets[:, 0]
            r = test_triplets[:, 1]
            o = test_triplets[:, 2]
            test_size = test_triplets.shape[0]
            num_ent = ent_mean.shape[0]
            o_mask = self.mask_eval_set(test_triplets, test_size, num_ent, time, mode="tail")
            s_mask = self.mask_eval_set(test_triplets, test_size, num_ent, time, mode="head")
            # perturb object
            ranks_o = self.perturb_and_get_rank(ent_mean, ent_mean_inv, rel_enc_means, s, r, o, test_size, o_mask, eval_bz, mode='tail')
            # perturb subject
            ranks_s = self.perturb_and_get_rank(ent_mean, ent_mean_inv, rel_enc_means, s, r, o, test_size, s_mask, eval_bz, mode='head')
            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1 # change to 1-indexed

            mrr = torch.mean(1.0 / ranks.float()).item()
            hit_1 = torch.mean((ranks <= 1).float()).item()
            hit_3 = torch.mean((ranks <= 3).float()).item()
            hit_10 = torch.mean((ranks <= 10).float()).item()

        return mrr, hit_1, hit_3, hit_10

    def perturb_and_get_rank(self, ent_mean, ent_mean_inv, rel_enc_means, s, r, o, test_size, mask, batch_size=100, mode ='tail'):
        """ Perturb one element in the triplets
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]

            if mode == 'tail':
                batch_s = ent_mean[s[batch_start: batch_end]]
                batch_o = ent_mean_inv
                target = o[batch_start: batch_end]
            else:
                batch_s = ent_mean
                batch_o = ent_mean_inv[o[batch_start: batch_end]]
                target = s[batch_start: batch_end]

            unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
            # import pdb; pdb.set_trace()
            masked_score = torch.where(mask[batch_start: batch_end], -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)
            score = torch.sigmoid(masked_score)  # bsz, n_ent

            ranks.append(self.sort_and_rank(score, target))
        return torch.cat(ranks)


class SimplE(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(SimplE, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.ent_embeds_inv = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds_inv = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))
        self.evaluater = SimpleEvaluationFilter(args, distmult, graph_dict_train, graph_dict_val, graph_dict_test)

        nn.init.xavier_uniform_(self.ent_embeds_inv, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds_inv, gain=nn.init.calculate_gain('relu'))

    def build_model(self):
        self.negative_rate = self.args.negative_rate

    def get_per_graph_ent_embeds(self, g_list):
        batched_graph = dgl.batch(g_list)
        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        ent_embeds_inv = self.ent_embeds_inv[batched_graph.ndata['id']].view(-1, self.embed_size)
        node_sizes = [len(g.nodes()) for g in g_list]
        return ent_embeds.split(node_sizes), ent_embeds_inv.split(node_sizes)

    def train_link_prediction(self, ent_embed, ent_embed_inv, triplets, neg_samples, labels, corrupt_tail=True):
        r = self.rel_embeds[triplets[:, 1]]
        r_inv = self.rel_embeds_inv[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            s_inv = ent_embed_inv[triplets[:, 0]]
            neg_o = ent_embed[neg_samples]
            neg_o_inv = ent_embed_inv[neg_samples]
            score = simple(s, s_inv, r, r_inv, neg_o, neg_o_inv, mode='tail')
        else:
            neg_s = ent_embed[neg_samples]
            neg_s_inv = ent_embed_inv[neg_samples]
            o = ent_embed[triplets[:, 2]]
            o_inv = ent_embed_inv[triplets[:, 2]]
            score = simple(neg_s, neg_s_inv, r, r_inv, o, o_inv, mode='head')
        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def link_classification_loss(self, ent_embed, ent_embed_inv, rel_embeds, triplets, labels):
        # triplets is a list of extrapolation samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = rel_embeds[triplets[:, 1]]
        o = ent_embed_inv[triplets[:, 2]]
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_list = [graph_dict[i.item()] for i in t_list]
        per_graph_ent_embeds, per_graph_ent_embeds_inv = self.get_per_graph_ent_embeds(g_list)
        triplets, labels = self.corrupter.sample_labels_val(g_list)
        return self.calc_metrics(per_graph_ent_embeds, per_graph_ent_embeds_inv, t_list, triplets, labels)

    def forward(self, t_list, reverse=False):
        kld_loss = 0
        reconstruct_loss = 0
        g_list = [self.graph_dict_train[i.item()] for i in t_list]
        per_graph_ent_embeds, per_graph_ent_embeds_inv = self.get_per_graph_ent_embeds(g_list)
        triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t_list, g_list)

        i = 0
        for ent_embed, ent_embed_inv in zip(per_graph_ent_embeds, per_graph_ent_embeds_inv):
            loss_tail = self.train_link_prediction(ent_embed, ent_embed_inv, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, ent_embed_inv, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss, kld_loss

    def calc_metrics(self, per_graph_ent_embeds, per_graph_ent_embeds_inv, t_list, triplets, labels):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        i = 0
        for ent_embed, ent_embed_inv in zip(per_graph_ent_embeds, per_graph_ent_embeds_inv):
            if triplets[i].shape[0] == 0: continue
            mrr, hit_1, hit_3, hit_10 = self.evaluater.calc_metrics_single_graph(ent_embed, ent_embed_inv, self.rel_embeds, triplets[i], t_list[i])
            loss = self.link_classification_loss(ent_embed, ent_embed_inv, self.rel_embeds, triplets[i], labels[i])
            mrrs.append(mrr)
            hit_1s.append(hit_1)
            hit_3s.append(hit_3)
            hit_10s.append(hit_10)
            losses.append(loss.item())
            i += 1
        return np.mean(mrrs), np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_10s), np.sum(losses)