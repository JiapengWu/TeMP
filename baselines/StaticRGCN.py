from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import reparametrize, move_dgl_to_cuda
from utils.scores import *
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent
from utils.evaluation import calc_metrics


class StaticRGCN(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(StaticRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.half_size = int(self.embed_size / 2)
        self.train_seq_len = self.args.train_seq_len
        self.test_seq_len = self.args.test_seq_len
        self.num_pos_facts = self.args.num_pos_facts

        self.ent_encoder = RGCN(self.args, self.hidden_size, self.embed_size, self.num_rels, static=True)
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.embed_size))
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels * 2, self.embed_size))

        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        graph_train_list = [self.graph_dict_train[i.item()] for i in t_list]
        g_list = [graph_dict[i.item()] for i in t_list]
        per_graph_ent_embeds = self.get_per_graph_ent_embeds(graph_train_list)
        triplets, labels = self.corrupter.sample_labels_val(g_list)
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        for i, ent_embed in enumerate(per_graph_ent_embeds):
            mrr, hit_1, hit_3, hit_10 = calc_metrics(ent_embed, self.rel_embeds, triplets[i])
            loss = self.link_classification_loss(ent_embed, self.rel_embeds, triplets[i], labels[i])
            mrrs.append(mrr)
            hit_1s.append(hit_1)
            hit_3s.append(hit_3)
            hit_10s.append(hit_10)
            losses.append(loss.item())
        return np.mean(mrrs), np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_10s), np.sum(losses), 0

    def forward(self, t_list, reverse=False):
        kld_loss = 0
        reconstruct_loss = 0
        g_list = [self.graph_dict_train[i.item()] for i in t_list]

        per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_list)
        triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.samples_labels_train(t_list, g_list)

        for i, ent_embed in enumerate(per_graph_ent_embeds):
            loss_tail = self.train_link_prediction(ent_embed, triplets[i], neg_tail_samples[i], labels[i], corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets[i], neg_head_samples[i], labels[i], corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
        # print("Graph reconstruction: {}".format(time.time() - start))
        return reconstruct_loss, kld_loss

    def get_per_graph_ent_embeds(self, graph_train_list):
        batched_graph = dgl.batch(graph_train_list)
        batched_graph.ndata['h'] = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        node_sizes = [len(g.nodes()) for g in graph_train_list]
        enc_ent_mean_graph = self.ent_encoder(batched_graph, reverse=False)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)
        return per_graph_ent_embeds

