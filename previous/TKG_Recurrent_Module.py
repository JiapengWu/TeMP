from torch import nn
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

    def get_val_vars(self, g_batched_list, t, h):
        g_batched_list_t = filter_none(g_batched_list[t])
        bsz = len(g_batched_list_t)
        triplets, labels = self.corrupter.sample_labels_val(g_batched_list_t)
        cur_h = h[-1][:bsz]  # bsz, hidden_size
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in g_batched_list_t]
        return g_batched_list_t, bsz, cur_h, triplets, labels, node_sizes

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

    def forward(self, t_list, reverse=False):
        kld_loss = 0
        reconstruct_loss = 0
        h = self.h0.expand(self.num_layers, len(t_list), self.hidden_size).contiguous()
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)

        for t in range(self.train_seq_len - 1):
            g_batched_list_t, bsz, cur_h, triplets, labels, node_sizes = self.get_val_vars(g_batched_list, t, h)
            # triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.samples_labels_train(time_batched_list_t, g_batched_list_t)
            per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, cur_h, node_sizes, val=True)

            pooled_fact_embeddings = []
            for i, ent_embed in enumerate(per_graph_ent_embeds):
                pooled_fact_embeddings.append(self.get_pooled_facts(ent_embed, triplets[i]))
            _, h = self.rnn(torch.stack(pooled_fact_embeddings, dim=0).unsqueeze(0), h[:, :bsz])

        train_graphs, time_batched_list_t = filter_none(g_batched_list[-1]), filter_none(time_batched_list[-1])
        bsz = len(train_graphs)
        cur_h = h[-1][:bsz]  # bsz, hidden_size
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t

        node_sizes = [len(g.nodes()) for g in train_graphs]
        triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(time_batched_list_t, train_graphs)
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graphs, cur_h, node_sizes)

        for i, ent_embed in enumerate(per_graph_ent_embeds):
            loss_tail = self.train_link_prediction(ent_embed, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        return reconstruct_loss, kld_loss
