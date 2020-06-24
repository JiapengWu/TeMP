from torch import nn
from models.TKG_Module import TKG_Module
from utils.utils import filter_none
import torch
from models.DRGCN import DRGCN
import numpy as np
from utils.utils import move_dgl_to_cuda, comp_deg_norm
from utils.scores import *
import dgl


class DRGCN_Module(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(DRGCN_Module, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)
        self.num_layers = self.args.num_layers
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len

        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def build_model(self):
        self.ent_encoder = DRGCN(self.args, self.hidden_size, self.embed_size, self.num_rels)

    def get_per_graph_ent_embeds(self, g_batched_list_t, node_sizes, prev_graph_embeds=None, val=False):
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
        batched_graph.ndata['h'] = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)

        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        enc_ent_mean_graph = self.ent_encoder(batched_graph, reverse=False)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)
        return per_graph_ent_embeds

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, graph_dict)

        for t in range(self.test_seq_len - 1):
            g_batched_list_t, bsz, cur_h, triplets, labels, node_sizes = self.get_val_vars(g_train_batched_list, t, h)
            per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, cur_h, node_sizes, val=True)

            pooled_fact_embeddings = []
            for i, ent_embed in enumerate(per_graph_ent_embeds):
                pooled_fact_embeddings.append(self.get_pooled_facts(ent_embed, triplets[i]))

            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])

        test_graph, bsz, cur_h, triplets, labels, _ = self.get_val_vars(g_batched_list, -1, h)
        train_graph = filter_none(g_train_batched_list[-1])
        node_sizes = [len(g.nodes()) for g in train_graph]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graph, cur_h, node_sizes, val=True)
        return self.calc_metrics(per_graph_ent_embeds, time_list[-1], triplets, labels)

    def get_val_vars(self, g_batched_list, t, h):
        g_batched_list_t = filter_none(g_batched_list[t])
        bsz = len(g_batched_list_t)
        samples, labels = self.corrupter.sample_labels_val(g_batched_list_t)
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in g_batched_list_t]
        return g_batched_list_t, bsz, samples, labels, node_sizes

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)

        for t in range(self.train_seq_len - 1):
            g_batched_list_t, bsz, samples, labels, node_sizes = self.get_val_vars(g_batched_list, t, h)
            # triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.samples_labels_train(time_batched_list_t, g_batched_list_t)
            per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, node_sizes, val=True)

            pooled_fact_embeddings = []
            for i, ent_embed in enumerate(per_graph_ent_embeds):
                pooled_fact_embeddings.append(self.get_pooled_facts(ent_embed, samples[i]))
            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])

        # TODO: fix the problem that when g_batched_list[-1] is None,
        #  the algorithm should still learn to predict the missing facts
        train_graphs, time_batched_list_t = filter_none(g_batched_list[-1]), filter_none(time_batched_list[-1])

        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in train_graphs]
        samples, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(time_batched_list_t, train_graphs)
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graphs, node_sizes)

        for i, ent_embed in enumerate(per_graph_ent_embeds):
            loss_tail = self.train_link_prediction(ent_embed, samples[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, samples[i], neg_head_samples[i], labels[i], corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        return reconstruct_loss
