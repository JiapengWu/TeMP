from torch import nn
import numpy as np
from utils.scores import *
from utils.evaluation import calc_metrics
from models.TKG_Module import TKG_Module
from utils.utils import filter_none
import torch


class TKG_Recurrent_Module(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Recurrent_Module, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
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

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, graph_dict)

        for t in range(self.test_seq_len - 1):
            g_batched_list_t, bsz, cur_h, triplets, labels, node_sizes = self.get_val_vars(g_train_batched_list, t, h)
            per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, cur_h, node_sizes)

            pooled_fact_embeddings = []
            for i, ent_embed in enumerate(per_graph_ent_embeds):
                pooled_fact_embeddings.append(self.get_pooled_facts(ent_embed, triplets[i]))

            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])
        # import pdb; pdb.set_trace()
        test_graph, bsz, cur_h, triplets, labels, node_sizes = self.get_val_vars(g_batched_list, -1, h)
        train_graph = filter_none(g_train_batched_list[-1])
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graph, cur_h, node_sizes)

        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        for i, ent_embed in enumerate(per_graph_ent_embeds):
            mrr, hit_1, hit_3, hit_10 = calc_metrics(ent_embed, self.rel_embeds, triplets[i])
            val_loss = self.link_classification_loss(ent_embed, self.rel_embeds, triplets[i], labels[i])
            mrrs.append(mrr)
            hit_1s.append(hit_1)
            hit_3s.append(hit_3)
            hit_10s.append(hit_10)
            losses.append(val_loss.item())

        return np.mean(mrrs), np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_10s), np.sum(losses)

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
            per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, cur_h, node_sizes)
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.samples_labels_train(time_batched_list_t, g_batched_list_t)

            pooled_fact_embeddings = []
            for i, ent_embed in enumerate(per_graph_ent_embeds):
                loss_tail = self.train_link_prediction(ent_embed, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
                loss_head = self.train_link_prediction(ent_embed, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
                pooled_fact_embeddings.append(self.get_pooled_facts(ent_embed, triplets[i]))
                reconstruct_loss += loss_tail + loss_head
            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])
        return reconstruct_loss, kld_loss
