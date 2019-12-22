import numpy as np
from utils.scores import *
import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import TimeDataset
from argparse import Namespace
from utils.CorrptTriples import CorruptTriples
import torch.nn.functional as F
from utils.utils import filter_none
from utils.evaluation import EvaluationFilter
from models.TKG_Module import TKG_Module


class Time_Dep_Module(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(Time_Dep_Module, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def _dataloader(self, times):
        # when using multi-node (ddp) we need to add the  datasampler
        dataset = TimeDataset(times)
        batch_size = self.args.batch_size
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        return self._dataloader(self.total_time)

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(self.total_time)

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.total_time)

    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, corrupt_tail=True):
        r = self.rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = ent_embed[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = ent_embed[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')
        # pdb.set_trace()
        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def link_classification_loss(self, ent_embed, rel_embeds, triplets, labels):
        # triplets is a list of extrapolation samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = rel_embeds[triplets[:, 1]]
        o = ent_embed[triplets[:, 2]]
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def get_pooled_facts(self, ent_embed, triples):
        s = ent_embed[triples[:, 0]]
        r = self.rel_embeds[triples[:, 1]]
        o = ent_embed[triples[:, 2]]
        pos_facts = torch.cat([s, r, o], dim=1)
        return torch.max(pos_facts, dim=0)[0]

    def get_val_vars(self, g_batched_list, t, h):
        g_batched_list_t = filter_none(g_batched_list[t])
        bsz = len(g_batched_list_t)
        triplets, labels = self.corrupter.sample_labels_val(g_batched_list_t)
        cur_h = h[-1][:bsz]  # bsz, hidden_size
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in g_batched_list_t]
        return g_batched_list_t, bsz, cur_h, triplets, labels, node_sizes

    def get_batch_graph_list(self, t_list, seq_len, graph_dict):
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]  # compute time unit
        time_list = []
        len_non_zero = []

        t_list = t_list.sort(descending=True)[0]
        g_list = []
        for tim in t_list:
            length = int(tim / time_unit) + 1
            cur_seq_len = seq_len if seq_len <= length else length
            time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
            time_list.append(time_seq + ([None] * (seq_len - len(time_seq))))
            len_non_zero.append(cur_seq_len)
            g_list.append([graph_dict[t] for t in time_seq] + ([None] * (seq_len - len(time_seq))))

        t_batched_list = [list(x) for x in zip(*time_list)]
        g_batched_list = [list(x) for x in zip(*g_list)]
        return g_batched_list, t_batched_list

    def calc_metrics(self, per_graph_ent_embeds, t_list, triplets, labels):

        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        for i, ent_embed in enumerate(per_graph_ent_embeds):
            if triplets[i].shape[0] == 0: continue
            mrr, hit_1, hit_3, hit_10 = self.evaluater.calc_metrics_single_graph(ent_embed, self.rel_embeds, triplets[i], t_list[i])
            loss = self.link_classification_loss(ent_embed, self.rel_embeds, triplets[i], labels[i])
            mrrs.append(mrr)
            hit_1s.append(hit_1)
            hit_3s.append(hit_3)
            hit_10s.append(hit_10)
            losses.append(loss.item())
        return np.mean(mrrs), np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_10s), np.sum(losses)
