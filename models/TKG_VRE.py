from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import samples_labels, cuda


class VKG_VRE(nn.Module):
    def __init__(self, args, num_ents, num_rels):

        super(VKG_VRE, self).__init__()
        self.args = args
        self.num_rels = num_rels
        self.negative_rate = self.args.negative_rate
        self.hidden_size = hidden_size = args.hidden_size
        self.embed_size = embed_size = args.embed_size
        self.num_layers = args.num_layers
        self.seq_len = self.args.seq_len
        self.use_cuda = args.use_cuda
        # encoder
        self.enc = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size),
            nn.ReLU())

        # self.ent_enc_means = nn.ModuleList([self.mean_encoder(hidden_size, embed_size)] * num_ents)
        # self.ent_enc_stds = nn.ModuleList([self.std_encoder(hidden_size, embed_size)] * num_ents)

        self.h0 = nn.Parameter(torch.Tensor(self.num_layers, 1, self.hidden_size))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, embed_size))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h0, gain=nn.init.calculate_gain('relu'))

        # prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

        self.ent_prior_means = self.mean_encoder(hidden_size, embed_size)
        self.ent_prior_stds = self.std_encoder(hidden_size, embed_size)

        self.rel_prior_means = self.mean_encoder(hidden_size, embed_size)
        self.rel_prior_stds = self.std_encoder(hidden_size, embed_size)

        self.ent_enc_means = RGCN(args, hidden_size, embed_size, num_rels)
        self.ent_enc_stds = RGCN(args, hidden_size, embed_size, num_rels)

        self.rel_enc_means = nn.ModuleList([self.mean_encoder(hidden_size, embed_size)] * num_rels*2)
        self.rel_enc_stds = nn.ModuleList([self.std_encoder(hidden_size, embed_size)] * num_rels*2)

        self.rnn = nn.GRU(input_size=embed_size * 3, hidden_size=hidden_size, num_layers=self.num_layers)

    def reparametrize(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        if self.use_cuda:
            eps = cuda(eps)
        return mean + (eps * std)

    @staticmethod
    def mean_encoder(hidden_size, embed_size):
        return nn.Linear(hidden_size, embed_size)

    @staticmethod
    def std_encoder(hidden_size, embed_size):
        return nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.Softplus())

    def get_relation_encoder_embeddings(self, h):
        bsz = h.shape[0]
        rel_enc_means = torch.zeros(2 * self.num_rels, bsz, self.embed_size)
        rel_enc_stds = torch.zeros(2 * self.num_rels, bsz, self.embed_size)
        for i in range(2 * self.num_rels):
            rel_enc_means[i] = self.rel_enc_means[i](h)
            rel_enc_stds[i] = self.rel_enc_stds[i](h)
        return rel_enc_means.transpose(0,1), rel_enc_stds.transpose(0,1)

    def calc_score(self, s,r,o):
        # DistMult
        # embedding are indexed by g.nodes()
        # triples are indexed by g.ndata['id']
        score = torch.sum(s * r * o, dim=1)
        return score

    def link_classification_loss(self, ent_mean, ent_std, rel_enc_means, rel_enc_stds, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        # import pdb; pdb.set_trace()
        s = self.reparametrize(ent_mean[triplets[:,0]], ent_std[triplets[:,0]])
        r = self.reparametrize(rel_enc_means[triplets[:,1]], rel_enc_stds[triplets[:,1]])
        o = self.reparametrize(ent_mean[triplets[:,2]], ent_std[triplets[:,2]])
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        pos_mask = labels==1
        pos_facts = torch.cat([s[pos_mask], r[pos_mask], o[pos_mask]], dim=1)
        return predict_loss, torch.max(pos_facts, dim=0)[0]

    def kld_gauss(self, q_mean, q_std, p_mean, p_std):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(p_std) - 2 * torch.log(q_std) + (q_std.pow(2) + (q_mean - p_mean).pow(2)) / p_std.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def forward(self, t_list, graph_dict, reverse=False):
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0] # compute time unit
        time_list = []
        len_non_zero = []

        t_list = t_list.sort(descending=True)[0]
        bsz = num_non_zero = len(torch.nonzero(t_list))
        t_list = t_list[:num_non_zero]

        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_list = []
        for tim in t_list:
            length = int(tim / time_unit) + 1
            seq_len = self.seq_len if self.seq_len <= length else length
            time_seq = times[length - self.seq_len:length] if self.seq_len <= length else times[:length]
            time_list.append(torch.LongTensor(time_seq))
            len_non_zero.append(seq_len)
            g_list.append([graph_dict[t] for t in time_seq] + ([None] * (self.seq_len - len(time_seq))))

        g_batched_list = [list(x) for x in zip(*g_list)]

        kld_loss = 0
        reconstruct_loss = 0
        for t in range(self.seq_len):
            triplets, labels = samples_labels(g_batched_list[t], self.negative_rate, self.use_cuda)
            batched_graph = dgl.batch(list(filter(lambda x: x is not None, g_batched_list[t])))

            # import pdb; pdb.set_trace()
            cur_h = h[-1]  # bsz, hidden_size

            # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
            node_sizes = [len(g.nodes()) if g else None for g in g_batched_list[t]]
            ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
            expanded_h = torch.cat([cur_h[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)
            batched_graph.ndata['h'] = torch.cat([ent_embeds, expanded_h], dim=-1)

            enc_ent_mean_graph = self.ent_enc_means(batched_graph, reverse)
            enc_ent_std_graph = self.ent_enc_stds(batched_graph, reverse)

            # sum_num_ents * hsz
            enc_ent_mean = enc_ent_mean_graph.ndata['h']
            enc_ent_std = enc_ent_std_graph.ndata['h']

            per_graph_ent_mean = enc_ent_mean.split(node_sizes)
            per_graph_ent_std = enc_ent_std.split(node_sizes)

            # extract triples, decode (s, r, o) for each triples and run negative sampling
            # bsz, 2* num_rels, hsz
            rel_enc_means, rel_enc_stds = self.get_relation_encoder_embeddings(cur_h)
            if self.use_cuda:
                rel_enc_means, rel_enc_stds = cuda(rel_enc_means), cuda(rel_enc_stds)
            # run distmult decoding
            i = 0
            pooled_fact_embeddings = []
            for ent_mean, ent_std in zip(per_graph_ent_mean, per_graph_ent_std):
                loss, pos_facts = self.link_classification_loss(ent_mean, ent_std, rel_enc_means[i], rel_enc_stds[i], triplets[i], labels[i])
                reconstruct_loss += loss
                pooled_fact_embeddings.append(pos_facts)
                i += 1
            # max pooling
            # pooled_ent_embeddings = torch.stack([torch.max(self.reparametrize(ent_mean, ent_std), dim=0)[0] for ent_mean, ent_std in zip(per_graph_ent_mean, per_graph_ent_std)], dim=0)
            # pooled_rel_embeddings = torch.stack([torch.max(self.reparametrize(rel_enc_means[i], rel_enc_stds[i]), dim=0)[0] for i in range(bsz)], dim=0)

            # get all the prior ent_embeddings and rel_embeddings in G_t

            # bsz, hsz
            prior_h = self.prior(cur_h)

            prior_ent_mean = self.ent_prior_means(prior_h)
            prior_ent_std = self.ent_prior_stds(prior_h)
            prior_rel_mean = self.rel_prior_means(prior_h)
            prior_rel_std = self.rel_prior_stds(prior_h)

            prior_ent_mean_extend = torch.cat([prior_ent_mean[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)
            prior_ent_std_extend = torch.cat([prior_ent_std[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)
            prior_rel_mean_extend = prior_rel_mean.unsqueeze(1).expand(bsz, 2 * self.num_rels, self.embed_size)
            prior_rel_std_extend = prior_rel_std.unsqueeze(1).expand(bsz, 2 * self.num_rels, self.embed_size)

            # compute loss

            kld_loss += self.kld_gauss(enc_ent_mean, enc_ent_std, prior_ent_mean_extend, prior_ent_std_extend)
            kld_loss += self.kld_gauss(rel_enc_means, rel_enc_stds, prior_rel_mean_extend, prior_rel_std_extend)
            # import pdb; pdb.set_trace()
            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h)
        return reconstruct_loss + kld_loss