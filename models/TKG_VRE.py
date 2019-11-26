from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import samples_labels, cuda, reparametrize
from utils.scores import *
from utils.evaluation import calc_metrics
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import TimeDataset


class TKG_VAE_Module(LightningModule):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_dev, graph_dict_test, train_times, valid_times, test_times):
        super(TKG_VAE_Module, self).__init__()
        # self.logger.log_hyperparams(args)
        self.graph_dict_train = graph_dict_train
        self.graph_dict_dev = graph_dict_dev
        self.graph_dict_test = graph_dict_test
        self.graph_dict_total = {**graph_dict_train, **graph_dict_dev, **graph_dict_test}
        pdb.set_trace()
        self.train_times = train_times
        self.valid_times = valid_times
        self.test_times = test_times
        self.build_model(args, num_rels, num_ents)

    def training_step(self, batch_time, batch_idx):
        reconstruct_loss, kld_loss = self.forward(batch_time)
        loss = reconstruct_loss + kld_loss
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'val_reconstruction_loss': reconstruct_loss,
            'val_KLD_loss': kld_loss,
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self.logger.experiment.log(output)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch_time, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        reconstruct_loss, kld_loss = self.forward(batch_time)
        loss = reconstruct_loss + kld_loss

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        output = OrderedDict({
            'val_reconstruction_loss': reconstruct_loss,
            'val_KLD_loss': kld_loss,
            'val_loss': loss,
        })
        self.logger.experiment.log(output)
        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch_time, batch_idx):
        pass

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0001)
        return optimizer

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
            num_workers=10
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        return self._dataloader(self.train_times)

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(self.valid_times)

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.test_times)

    def build_model(self, args, num_rels, num_ents):
        self.args = args
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.negative_rate = self.args.negative_rate
        self.hidden_size = hidden_size = args.hidden_size
        self.embed_size = embed_size = args.embed_size
        self.half_size = int(self.embed_size/2)
        self.num_layers = args.num_layers
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len
        self.use_cuda = args.use_cuda
        self.num_pos_facts = args.num_pos_facts
        self.use_VAE = self.args.use_VAE
        self.use_rgcn = self.args.use_rgcn
        # encoder
        self.enc = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size),
            nn.ReLU())
        self.calc_score = {'distmult': distmult, 'complex': complex}[self.args.score_function]

        # self.ent_enc_means = nn.ModuleList([self.mean_encoder(hidden_size, embed_size)] * num_ents)
        # self.ent_enc_stds = nn.ModuleList([self.std_encoder(hidden_size, embed_size)] * num_ents)

        self.h0 = nn.Parameter(torch.Tensor(self.num_layers, 1, self.hidden_size))
        self.ent_embeds = nn.Embedding(num_ents, embed_size)

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h0, gain=nn.init.calculate_gain('relu'))

        # prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        if self.use_VAE:
            self.ent_prior_means = self.mean_encoder(hidden_size, embed_size)
            self.ent_prior_stds = self.std_encoder(hidden_size, embed_size)

            self.rel_prior_means = self.mean_encoder(hidden_size, embed_size)
            self.rel_prior_stds = self.std_encoder(hidden_size, embed_size)

        if self.use_rgcn:
            self.ent_enc_means = RGCN(args, hidden_size, embed_size, num_rels)
            self.ent_enc_stds = RGCN(args, hidden_size, embed_size, num_rels)
        else:
            self.ent_enc_means = nn.ModuleList([self.mean_encoder(hidden_size, embed_size)] * num_ents)
            self.ent_enc_stds = nn.ModuleList([self.std_encoder(hidden_size, embed_size)] * num_ents)

        self.rel_enc_means = nn.ModuleList([self.mean_encoder(hidden_size, embed_size)] * num_rels * 2)
        self.rel_enc_stds = nn.ModuleList([self.std_encoder(hidden_size, embed_size)] * num_rels * 2)
        # self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, embed_size))
        # self.rel_enc_means = nn.Embedding(num_rels * 2, embed_size)
        # self.rel_enc_stds = nn.Embedding(num_rels * 2, embed_size)

        self.rnn = nn.GRU(input_size=embed_size * 3, hidden_size=hidden_size, num_layers=self.num_layers)

    @staticmethod
    def mean_encoder(hidden_size, embed_size):
        return nn.Linear(hidden_size, embed_size)

    @staticmethod
    def std_encoder(hidden_size, embed_size):
        return nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.Softplus())

    def link_classification_loss(self, ent_mean, ent_std, rel_enc_means, rel_enc_stds, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        # import pdb; pdb.set_trace()
        s = reparametrize(ent_mean[triplets[:,0]], ent_std[triplets[:,0]], self.use_cuda)
        r = reparametrize(rel_enc_means[triplets[:,1]], rel_enc_stds[triplets[:,1]], self.use_cuda)
        o = reparametrize(ent_mean[triplets[:,2]], ent_std[triplets[:,2]], self.use_cuda)
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        pos_mask = (labels==1)
        pos_facts = torch.cat([s[pos_mask], r[pos_mask], o[pos_mask]], dim=1)
        return predict_loss, torch.max(pos_facts, dim=0)[0]

    def kld_gauss(self, q_mean, q_std, p_mean, p_std):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(p_std) - 2 * torch.log(q_std) + (q_std.pow(2) + (q_mean - p_mean).pow(2)) / p_std.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    @staticmethod
    def filter_none(l):
        return list(filter(lambda x: x is not None, l))

    def get_batch_graph_list(self, t_list, seq_len):
        graph_dict = self.graph_dict_total
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]  # compute time unit
        time_list = []
        len_non_zero = []

        t_list = t_list.sort(descending=True)[0]
        # start = times[0]
        num_non_zero = len(torch.nonzero(t_list))
        t_list = t_list[:num_non_zero]

        g_list = []
        for tim in t_list:
            length = int(tim / time_unit) + 1
            cur_seq_len = seq_len if seq_len <= length else length
            time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
            time_list.append(torch.LongTensor(time_seq))
            len_non_zero.append(cur_seq_len)
            g_list.append([graph_dict[t] for t in time_seq] + ([None] * (seq_len - len(time_seq))))

        g_batched_list = [list(x) for x in zip(*g_list)]
        return g_batched_list

    def get_posterior_embeddings(self, g_batched_list_t, cur_h, node_sizes, reverse, val=False):
        bsz = len(g_batched_list_t)
        batched_graph = dgl.batch(g_batched_list_t)

        # sum_num_ents, bsz, hsz
        # import pdb; pdb.set_trace()
        expanded_h = torch.cat(
            [cur_h[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)
        # if self.use_rgcn:
        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        batched_graph.ndata['h'] = torch.cat([ent_embeds, expanded_h], dim=-1)

        enc_ent_mean_graph = self.ent_enc_means(batched_graph, reverse)
        ent_enc_means = enc_ent_mean_graph.ndata['h']
        per_graph_ent_mean = ent_enc_means.split(node_sizes)
        if not val:
            enc_ent_std_graph = self.ent_enc_stds(batched_graph, reverse)
            ent_enc_stds = F.softplus(enc_ent_std_graph.ndata['h'])
            per_graph_ent_std = ent_enc_stds.split(node_sizes)
        else:
            # import pdb; pdb.set_trace()
            idx = batched_graph.ndata['id'].squeeze()
            ent_enc_means = torch.zeros(len(batched_graph.nodes()), self.embed_size)
            ent_enc_stds = torch.zeros(len(batched_graph.nodes()), self.embed_size)
            for i in range(len(batched_graph.nodes())):
                ent_enc_means[i] = self.ent_enc_means[idx[i]](expanded_h[i])
                ent_enc_stds[i] = self.ent_enc_stds[idx[i]](expanded_h[i])
            # bsz, num_ents, hsz
            if self.use_cuda:
                ent_enc_means, ent_enc_stds = cuda(ent_enc_means), cuda(ent_enc_stds)

        # extract triples, decode (s, r, o) for each triples and run negative sampling
        # get enc relation embeddings
        rel_enc_means = torch.zeros(2 * self.num_rels, bsz, self.embed_size)
        for i in range(2 * self.num_rels):
            # bsz, hsz
            rel_enc_means[i] = self.rel_enc_means[i](cur_h)
        # bsz, 2 * num_rels, hsz
        rel_enc_means = rel_enc_means.transpose(0, 1)
        if self.use_cuda:
            rel_enc_means = cuda(rel_enc_means)
        if not val:
            rel_enc_stds = torch.zeros(2 * self.num_rels, bsz, self.embed_size)
            for i in range(2 * self.num_rels):
                rel_enc_stds[i] = self.rel_enc_stds[i](cur_h)
            rel_enc_stds = rel_enc_stds.transpose(0, 1)
            if self.use_cuda:
                rel_enc_stds = cuda(rel_enc_stds)

            # bsz, hsz
        if not val:
            return per_graph_ent_mean, per_graph_ent_std, rel_enc_means, rel_enc_stds, ent_enc_means, ent_enc_stds
        else:
            return per_graph_ent_mean, rel_enc_means, ent_enc_means

    def get_prior_embeddings(self, node_sizes, cur_h, bsz):
        prior_h = self.prior(cur_h)
        # bsz, hsz
        prior_ent_mean = self.ent_prior_means(prior_h)
        prior_ent_std = self.ent_prior_stds(prior_h)
        prior_rel_mean = self.rel_prior_means(prior_h)
        prior_rel_std = self.rel_prior_stds(prior_h)
        # sum_n_ent, bsz, hsz
        prior_ent_mean_extend = torch.cat(
            [prior_ent_mean[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)
        prior_ent_std_extend = torch.cat(
            [prior_ent_std[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)
        # bsz, 2 * num_rels, hsz
        prior_rel_mean_extend = prior_rel_mean.unsqueeze(1).expand(bsz, 2 * self.num_rels, self.embed_size)
        prior_rel_std_extend = prior_rel_std.unsqueeze(1).expand(bsz, 2 * self.num_rels, self.embed_size)
        return prior_ent_mean_extend, prior_ent_std_extend, prior_rel_mean_extend, prior_rel_std_extend

    def evaluate(self, t_list, graph_dict, reverse=False):
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_batched_list = self.get_batch_graph_list(t_list, self.test_seq_len)
        for t in range(self.test_seq_len - 1):
            g_batched_list_t = self.filter_none(g_batched_list[t])
            bsz = len(g_batched_list_t)
            triplets, labels = samples_labels(g_batched_list_t, self.negative_rate, self.use_cuda, self.num_pos_facts, val=True)
            cur_h = h[-1][:bsz]  # bsz, hidden_size
            # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
            node_sizes = [len(g.nodes()) for g in g_batched_list_t]
            per_graph_ent_mean, rel_enc_means, ent_enc_means = \
                self.get_posterior_embeddings(g_batched_list_t, cur_h, node_sizes, reverse)

            pooled_fact_embeddings = []

            i = 0
            for ent_mean in per_graph_ent_mean:
                s = triplets[i][:, 0]
                r = triplets[i][:, 1]
                o = triplets[i][:, 2]
                pos_facts = torch.cat([ent_mean[s], rel_enc_means[r], ent_mean[o]], dim=1)
                pooled_fact_embeddings.append(pos_facts)

            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])

    def forward(self, t_list, reverse=False):
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len)
        # pdb.set_trace()
        kld_loss = torch.tensor(0)
        reconstruct_loss = 0
        for t in range(self.train_seq_len):
            g_batched_list_t = self.filter_none(g_batched_list[t])
            bsz = len(g_batched_list_t)
            triplets, labels = samples_labels(g_batched_list_t, self.negative_rate, self.use_cuda, self.num_pos_facts)
            cur_h = h[-1][:bsz]  # bsz, hidden_size
            # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
            node_sizes = [len(g.nodes()) for g in g_batched_list_t]
            per_graph_ent_mean, per_graph_ent_std, rel_enc_means, rel_enc_stds, ent_enc_means, ent_enc_stds = \
                self.get_posterior_embeddings(g_batched_list_t, cur_h, node_sizes, reverse)

            # run distmult decoding
            i = 0
            pooled_fact_embeddings = []
            for ent_mean, ent_std in zip(per_graph_ent_mean, per_graph_ent_std):
                loss, pos_facts = self.link_classification_loss(ent_mean, ent_std, rel_enc_means[i], rel_enc_stds[i],
                                                                triplets[i], labels[i])
                reconstruct_loss += loss
                pooled_fact_embeddings.append(pos_facts)
                i += 1
            # get all the prior ent_embeddings and rel_embeddings in G_t
            # bsz, hsz
            if self.use_VAE :
                prior_ent_mean_extend, prior_ent_std_extend, prior_rel_mean_extend, prior_rel_std_extend = \
                    self.get_prior_embeddings(node_sizes, cur_h, bsz)
                # compute loss
                kld_loss += self.kld_gauss(ent_enc_means, ent_enc_stds, prior_ent_mean_extend, prior_ent_std_extend)
                kld_loss += self.kld_gauss(rel_enc_means, rel_enc_stds, prior_rel_mean_extend, prior_rel_std_extend)

            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])
        return reconstruct_loss, kld_loss