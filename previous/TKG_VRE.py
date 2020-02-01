from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import reparametrize, move_dgl_to_cuda, filter_none, comp_deg_norm
from utils.scores import *
from previous.TKG_Recurrent_Module import TKG_Recurrent_Module

class TKG_VAE(TKG_Recurrent_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_VAE, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.half_size = int(self.embed_size / 2)
        self.use_VAE = self.args.use_VAE

        if self.use_VAE:
            self.init_mean = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
            self.last_mean = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))

            self.prior = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU())
            # self.ent_prior_means = nn.Parameter(torch.zeros(self.num_ents, self.embed_size), requires_grad=False)
            # self.ent_prior_stds = nn.Parameter(torch.ones(self.num_ents, self.embed_size), requires_grad=False)
            # self.ent_prior_stds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
            # nn.init.xavier_uniform_(self.ent_prior_stds, gain=nn.init.calculate_gain('relu'))

            # self.rel_prior_means = nn.Parameter(torch.zeros(self.num_rels * 2, self.embed_size), requires_grad=False)
            # self.rel_prior_std = nn.Parameter(torch.ones(self.num_rels * 2, self.embed_size), requires_grad=False)
            # self.rel_prior_std = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))
            # nn.init.xavier_uniform_(self.rel_prior_std, gain=nn.init.calculate_gain('relu'))

            self.ent_prior_means = nn.ModuleList([self.mean_encoder(self.hidden_size, self.embed_size)] * self.num_ents)
            self.ent_prior_stds = nn.ModuleList([self.std_encoder(self.hidden_size, self.embed_size)] * self.num_ents)

            self.rel_prior_means = nn.Parameter(torch.zeros(self.num_rels * 2, self.embed_size), requires_grad=False)
            self.rel_prior_std = nn.Parameter(torch.ones(self.num_rels * 2, self.embed_size), requires_grad=False)

        self.rel_enc_stds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))
        nn.init.xavier_uniform_(self.rel_enc_stds, gain=nn.init.calculate_gain('relu'))
        self.ent_enc_means = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)
        self.ent_enc_stds = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)

    @staticmethod
    def mean_encoder(hidden_size, embed_size):
        return nn.Linear(hidden_size, embed_size)

    @staticmethod
    def std_encoder(hidden_size, embed_size):
        return nn.Sequential(
            nn.Linear(hidden_size, embed_size),
            nn.Softplus())

    def train_reparametrize_link_prediction(self, ent_mean, ent_std, triplets, neg_samples, labels, corrupt_tail=True):
        # r = reparametrize(self.rel_embeds[triplets[:, 1]], ent_mean.new_zeros(triplets.shape[0], self.embed_size))
        if self.use_VAE:
            r = reparametrize(self.rel_embeds[triplets[:, 1]], F.softplus(self.rel_enc_stds[triplets[:, 1]]))
        else:
            r = reparametrize(self.rel_embeds[triplets[:, 1]], ent_mean.new_zeros(triplets.shape[0], self.embed_size))
        if corrupt_tail:
            s = reparametrize(ent_mean[triplets[:,0]], ent_std[triplets[:,0]])
            neg_o = reparametrize(ent_mean[neg_samples], ent_std[neg_samples])
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = reparametrize(ent_mean[neg_samples], ent_std[neg_samples])
            o = reparametrize(ent_mean[triplets[:,2]], ent_std[triplets[:,2]])
            score = self.calc_score(neg_s, r, o, mode='head')

        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def kld_gauss(self, q_mean, q_std, p_mean, p_std):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(p_std) - 2 * torch.log(q_std) +
                       (q_std.pow(2) + (q_mean - p_mean).pow(2)) /
                       p_std.pow(2) - 1)
        return 0.5 / q_mean.shape[0] * torch.mean(torch.sum(kld_element, 1))

    def get_batch_graph(self, g_batched_list_t, cur_h, node_sizes, val):
        if val:
            sampled_graph_list = g_batched_list_t
        else:
            sampled_graph_list = []
            for g in g_batched_list_t:
                src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
                half_num_nodes = int(src.shape[0] / 2)
                graph_split_ids = np.random.choice(np.arange(half_num_nodes),
                                                   size=int(0.5 * half_num_nodes), replace=False)
                graph_split_rev_ids = graph_split_ids + half_num_nodes

                sg = g.edge_subgraph(np.concatenate((graph_split_ids, graph_split_rev_ids)), preserve_nodes=True)
                norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(norm).view(-1, 1)})
                sg.edata['type_s'] = rel[np.concatenate((graph_split_ids, graph_split_rev_ids))]
                sg.ids = g.ids
                sampled_graph_list.append(sg)

        batched_graph = dgl.batch(sampled_graph_list)
        # sum_num_ents, bsz, hsz
        expanded_h = torch.cat(
            [cur_h[i].unsqueeze(0).expand(size, self.embed_size) for i, size in enumerate(node_sizes)], dim=0)

        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        batched_graph.ndata['h'] = torch.cat([ent_embeds, expanded_h], dim=-1)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        return batched_graph

    def get_posterior_embeddings(self, g_batched_list_t, cur_h, node_sizes, val=False):
        batched_graph = self.get_batch_graph(g_batched_list_t, cur_h, node_sizes, val)
        enc_ent_mean_graph = self.ent_enc_means(batched_graph, reverse=False)
        ent_enc_means = enc_ent_mean_graph.ndata['h']
        per_graph_ent_mean = ent_enc_means.split(node_sizes)
        if self.use_VAE:
            enc_ent_std_graph = self.ent_enc_stds(batched_graph, reverse=False)
            ent_enc_stds = F.softplus(enc_ent_std_graph.ndata['h'])
            # ent_enc_stds = self.ent_enc_stds(batched_graph.ndata['h'])
            per_graph_ent_std = ent_enc_stds.split(node_sizes)
        else:
            per_graph_ent_std = ent_enc_stds = [None] * sum(node_sizes)
        return per_graph_ent_mean, per_graph_ent_std, ent_enc_means, ent_enc_stds

    def get_per_graph_ent_embeds(self, g_batched_list_t, cur_h, node_sizes, val=False):
        batched_graph = self.get_batch_graph(g_batched_list_t, cur_h, node_sizes, val)
        enc_ent_mean_graph = self.ent_enc_means(batched_graph, reverse=False)
        ent_enc_means = enc_ent_mean_graph.ndata['h']

        per_graph_ent_mean = ent_enc_means.split(node_sizes)
        return per_graph_ent_mean

    # def get_prior_from_hidden(self, g_batched_list_t, node_sizes, cur_h):
    #     batched_graph = dgl.batch(g_batched_list_t)
    #     prior_ent_means = self.ent_prior_means[batched_graph.ndata['id']].view(-1, self.embed_size)
    #     prior_ent_stds = F.softplus(self.ent_prior_stds[batched_graph.ndata['id']].view(-1, self.embed_size))
    #     return prior_ent_means, prior_ent_stds

    def get_prior_from_hidden(self, g_batched_list_t, node_sizes, cur_h):
        # bsz, hsz
        prior_h = self.prior(cur_h)
        # sum_n_ent, bsz, esz
        prior_ent_means_lst = []
        prior_ent_stds_lst = []
        for i, g in enumerate(g_batched_list_t):
            # n_ents, ndim
            prior_ent_mean = torch.stack([self.ent_prior_means[j](prior_h[i]) for j in g.ndata['id']], dim=0)
            prior_ent_std = torch.stack([self.ent_prior_stds[j](prior_h[i]) for j in g.ndata['id']], dim=0)

            prior_ent_means_lst.append(prior_ent_mean)
            prior_ent_stds_lst.append(prior_ent_std)

        prior_ent_means = torch.cat(prior_ent_means_lst, dim=0)
        prior_ent_stds = torch.cat(prior_ent_stds_lst, dim=0)

        return prior_ent_means, prior_ent_stds

    def forward(self, t_list, reverse=False):
        kld_loss = 0
        reconstruct_loss = 0
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)

        for t in range(self.train_seq_len):
            g_batched_list_t, time_batched_list_t = filter_none(g_batched_list[t]), filter_none(time_batched_list[t])
            bsz = len(g_batched_list_t)
            cur_h = h[-1][:bsz]  # bsz, hidden_size
            # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t

            node_sizes = [len(g.nodes()) for g in g_batched_list_t]
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(time_batched_list_t, g_batched_list_t)

            per_graph_ent_mean, per_graph_ent_std, ent_enc_means, ent_enc_stds = \
                self.get_posterior_embeddings(g_batched_list_t, cur_h, node_sizes)
            # run distmult decoding
            pooled_fact_embeddings = []
            i = 0
            for ent_mean, ent_std in zip(per_graph_ent_mean, per_graph_ent_std):
                if self.use_VAE:
                    loss_tail = self.train_reparametrize_link_prediction(ent_mean, ent_std, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
                    loss_head = self.train_reparametrize_link_prediction(ent_mean, ent_std, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
                else:
                    # loss_tail = self.train_reparametrize_link_prediction(ent_mean, ent_mean.new_zeros(ent_mean.shape), triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
                    # loss_head = self.train_reparametrize_link_prediction(ent_mean, ent_mean.new_zeros(ent_mean.shape), triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
                    loss_tail = self.train_link_prediction(ent_mean, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
                    loss_head = self.train_link_prediction(ent_mean, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)

                pooled_fact_embeddings.append(self.get_pooled_facts(ent_mean, triplets[i]))
                reconstruct_loss += loss_tail + loss_head
                i += 1

            # get all the prior ent_embeddings and rel_embeddings in G_t
            if self.use_VAE :
                prior_ent_means, prior_ent_stds = self.get_prior_from_hidden(g_batched_list_t, node_sizes, cur_h)
                kld_loss += self.kld_gauss(ent_enc_means, ent_enc_stds, prior_ent_means, prior_ent_stds)
                kld_loss += self.kld_gauss(self.rel_embeds, F.softplus(self.rel_enc_stds), self.rel_prior_means, self.rel_prior_std)

            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])
        return reconstruct_loss, kld_loss