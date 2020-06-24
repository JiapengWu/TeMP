from models.PostDynamicRGCN import ImputeDynamicRGCN, PostDynamicRGCN, PostEnsembleDynamicRGCN
import torch
import torch.nn as nn
from torch import nn
from models.TKG_Module import TKG_Module
from utils.utils import filter_none
import torch
from models.RRGCN import RRGCN
import numpy as np
from utils.utils import move_dgl_to_cuda, comp_deg_norm, node_norm_to_edge_norm, cuda
from utils.scores import *
import dgl
from utils.DropEdge import DropEdge
import torch.nn.functional as F
from utils.CorrptTriples import CorruptTriples
from models.BiDynamicRGCN import BiDynamicRGCN
from utils.post_evaluation import PostEvaluationFilter
from utils.post_evaluation import PostEnsembleEvaluationFilter
import pdb
from utils.evaluation import EvaluationFilter


class ImputeBiDynamicRGCN(BiDynamicRGCN, ImputeDynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=EvaluationFilter):
        ImputeDynamicRGCN.__init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)

    def build_model(self):
        BiDynamicRGCN.build_model(self)

    def get_all_embeds_Gt(self, convoluted_embeds, g, t, second_embeds_forward_loc, first_prev_graph_embeds_forward_rec, second_prev_graph_embeds_forward_rec, time_diff_tensor_forward,
                          second_embeds_backward_loc, first_prev_graph_embeds_backward_rec, second_prev_graph_embeds_backward_rec, time_diff_tensor_backward):
        all_embeds_g = self.ent_encoder.forward_isolated_impute(self.ent_embeds, first_prev_graph_embeds_forward_rec, second_prev_graph_embeds_forward_rec, time_diff_tensor_forward.unsqueeze(-1),
                                                                          first_prev_graph_embeds_backward_rec, second_prev_graph_embeds_backward_rec, time_diff_tensor_backward.unsqueeze(-1), t, second_embeds_forward_loc, second_embeds_backward_loc)

        keys = np.array(list(g.ids.keys()))
        values = np.array(list(g.ids.values()))

        all_embeds_g[values] = convoluted_embeds[keys]
        return all_embeds_g

    def get_per_graph_ent_dropout_embeds_one_direction(self, cur_time_list, target_time_list, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward):
        batched_graph = self.get_batch_graph_dropout_embeds(filter_none(cur_time_list), target_time_list)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        second_local_embeds, first_layer_embeds, second_layer_embeds = self.ent_encoder.forward_post_ensemble_one_direction(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, cur_time_list, node_sizes, forward)

        return second_local_embeds.split(node_sizes), first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_per_graph_ent_embeds_one_direction(self, g_batched_list_t, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward, full, rate=0.5):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full=full, rate=rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        second_local_embeds, first_layer_embeds, second_layer_embeds = self.ent_encoder.forward_post_ensemble_one_direction(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward)

        return second_local_embeds.split(node_sizes), first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_final_graph_embeds(self, g_batched_list_t, time_batched_list_t, seq_len, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                                    start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full):
        first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward = self.get_prev_embeddings(g_batched_list_t, hist_embeddings_forward_rec, start_time_tensor_forward, seq_len - 1)
        first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward = self.get_prev_embeddings(g_batched_list_t, hist_embeddings_backward_rec, start_time_tensor_backward, seq_len - 1)

        node_sizes = [len(g.nodes()) for g in g_batched_list_t]

        return self.get_graph_embeds_center(g_batched_list_t, time_batched_list_t, node_sizes, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                            first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, full=full)

    def get_graph_embeds_center(self, g_batched_list_t, time_batched_list_t, node_sizes, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                               first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, full, rate=0.5):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        second_local_embeds, second_layer_embeds = self.ent_encoder.forward_post_ensemble(batched_graph, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                                 first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes)

        return second_local_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def pre_forward(self, g_batched_list, time_batched_list, forward=True, val=False):
        seq_len = self.test_seq_len if val else self.train_seq_len
        bsz = len(g_batched_list[0])
        target_time_batched_list = time_batched_list[-1]
        hist_embeddings_loc = self.ent_embeds.new_zeros(bsz, self.num_ents, self.embed_size)
        hist_embeddings_rec = self.ent_embeds.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        start_time_tensor = self.ent_embeds.new_zeros(bsz, self.num_ents)
        full = val or not self.args.random_dropout
        for cur_t in range(seq_len - 1):
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, cur_t)

            if len(g_batched_list_t) == 0: continue
            first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(g_batched_list_t, hist_embeddings_rec, start_time_tensor, cur_t)
            if self.edge_dropout and not val:
                second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_dropout_embeds_one_direction(time_batched_list[cur_t], target_time_batched_list, node_sizes,
                                                                                                                  time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward)
            else:
                second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds_one_direction(g_batched_list_t, time_batched_list[cur_t], node_sizes,
                                                                                                                  time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward, full=full, rate=0.8)
            hist_embeddings_loc, hist_embeddings_rec = self.update_time_diff_hist_embeddings(second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, cur_t, bsz)
        if not forward:
            hist_embeddings_loc = torch.flip(hist_embeddings_loc, [0])
            hist_embeddings_rec = torch.flip(hist_embeddings_rec, [0])
            start_time_tensor = torch.flip(start_time_tensor, [0])
        return hist_embeddings_loc, hist_embeddings_rec, start_time_tensor

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True)
        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False)

        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        _, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.train_seq_len, hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward,
                                                              hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full=False)

        i = 0
        for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_rec):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            time_diff_tensor_forward = self.train_seq_len - 1 - start_time_tensor_forward[i]
            time_diff_tensor_backward = self.train_seq_len - 1 - start_time_tensor_backward[i]
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_forward_loc[i], hist_embeddings_forward_rec[i][0], hist_embeddings_forward_rec[i][1], time_diff_tensor_forward,
                                                                   hist_embeddings_backward_loc[i], hist_embeddings_backward_rec[i][0], hist_embeddings_backward_rec[i][1], time_diff_tensor_backward)
            loss_tail = BiDynamicRGCN.train_link_prediction(self, ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = BiDynamicRGCN.train_link_prediction(self, ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate_embed(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(
            t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list, _, _ = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward = self.pre_forward(
            g_forward_batched_list, t_forward_batched_list, forward=True, val=True)
        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.pre_forward(
            g_backward_batched_list, t_backward_batched_list, forward=False, val=True)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        _, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs,time_batched_list_t, self.test_seq_len, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                                        start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full=True)

        return per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward

    def evaluate(self, t_list, val=True):
        per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec, \
        start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward,
                                                                        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, self.test_seq_len - 1)

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list, hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward,
                                                    hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):
            time_diff_tensor_forward = cur_t - start_time_tensor_forward[i]
            time_diff_tensor_backward = cur_t - start_time_tensor_backward[i]
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_forward_loc[i], hist_embeddings_forward_rec[i][0], hist_embeddings_forward_rec[i][1], time_diff_tensor_forward,
                                                                hist_embeddings_backward_loc[i], hist_embeddings_backward_rec[i][0], hist_embeddings_backward_rec[i][1], time_diff_tensor_backward)
            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                label = cuda(label)
            if index_sample.shape[0] == 0: continue
            rank = self.evaluater.calc_metrics_single_graph(ent_embed, self.rel_embeds, all_embeds_g, index_sample, g, t)
            loss = self.link_classification_loss(ent_embed, self.rel_embeds, index_sample, label)
            ranks.append(rank)
            losses.append(loss.item())
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)


class PostBiDynamicRGCN(ImputeBiDynamicRGCN, PostDynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=PostEvaluationFilter):
        super(PostBiDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)
        self.drop_edge = DropEdge(args, graph_dict_train, graph_dict_val, graph_dict_test)
        self.init_freq_mlp()

    def get_all_embeds_Gt(self, convoluted_embeds_loc, convoluted_embeds_rec, g, t, second_embeds_forward_loc, first_prev_graph_embeds_forward_rec, second_prev_graph_embeds_forward_rec, time_diff_tensor_forward,
                          second_embeds_backward_loc, first_prev_graph_embeds_backward_rec, second_prev_graph_embeds_backward_rec, time_diff_tensor_backward):
        all_embeds_g_loc, all_embeds_g_rec = self.ent_encoder.forward_post_ensemble_isolated(self.ent_embeds, first_prev_graph_embeds_forward_rec, second_prev_graph_embeds_forward_rec, time_diff_tensor_forward.unsqueeze(-1),
                                                                          first_prev_graph_embeds_backward_rec, second_prev_graph_embeds_backward_rec, time_diff_tensor_backward.unsqueeze(-1), t, second_embeds_forward_loc, second_embeds_backward_loc)

        keys = np.array(list(g.ids.keys()))
        values = np.array(list(g.ids.values()))
        all_embeds_g_loc[values] = convoluted_embeds_loc[keys]
        all_embeds_g_rec[values] = convoluted_embeds_rec[keys]
        # for k, v in g.ids.items():
        #     all_embeds_g_loc[v] = convoluted_embeds_loc[k]
        #     all_embeds_g_rec[v] = convoluted_embeds_rec[k]
        return all_embeds_g_loc, all_embeds_g_rec

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True)
        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False)

        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.train_seq_len, hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward,
                                                              hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full=False)

        i = 0
        for t, g, ent_embed_loc, ent_embed_rec in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            time_diff_tensor_forward = self.train_seq_len - 1 - start_time_tensor_forward[i]
            time_diff_tensor_backward = self.train_seq_len - 1 - start_time_tensor_backward[i]
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_forward_loc[i], hist_embeddings_forward_rec[i][0], hist_embeddings_forward_rec[i][1], time_diff_tensor_forward,
                                                                   hist_embeddings_backward_loc[i], hist_embeddings_backward_rec[i][0], hist_embeddings_backward_rec[i][1], time_diff_tensor_backward)
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed = self.calc_ensemble_ratio(triplets, t, g)
            loss_tail = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_tail_samples, labels, all_embeds_g_loc,
                                                   all_embeds_g_rec, weight_object_query_subject_embed, weight_object_query_object_embed, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_head_samples, labels, all_embeds_g_loc,
                                                   all_embeds_g_rec, weight_subject_query_subject_embed, weight_subject_query_object_embed, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate_embed(self, t_list, val=True):

        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(
            t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list, _, _ = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward = self.pre_forward(
            g_forward_batched_list, t_forward_batched_list, forward=True, val=True)
        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.pre_forward(
            g_backward_batched_list, t_backward_batched_list, forward=False, val=True)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs,time_batched_list_t, self.test_seq_len, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                                        start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full=True)
        return per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec, \
                                        start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward

    def evaluate(self, t_list, val=True):
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec,\
            start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                                        start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, self.test_seq_len - 1)

    def calc_metrics(self, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, g_list, t_list, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                     start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed_loc, ent_embed_rec in zip(g_list, t_list, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            time_diff_tensor_forward = cur_t - start_time_tensor_forward[i]
            time_diff_tensor_backward = cur_t - start_time_tensor_backward[i]
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_forward_loc[i], hist_embeddings_forward_rec[i][0], hist_embeddings_forward_rec[i][1],
                                                                         time_diff_tensor_forward, hist_embeddings_backward_loc[i], hist_embeddings_backward_rec[i][0], hist_embeddings_backward_rec[i][1], time_diff_tensor_backward)

            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            # label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                # label = cuda(label)
            if index_sample.shape[0] == 0: continue
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed = self.calc_ensemble_ratio(index_sample, t, g)
            # pdb.set_trace()
            rank = self.evaluater.calc_metrics_single_graph(ent_embed_loc, ent_embed_rec, self.rel_embeds, all_embeds_g_loc, all_embeds_g_rec, index_sample, weight_subject_query_subject_embed,
                                                            weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed, g, t)

            # loss = self.link_classification_loss(ent_embed_loc, ent_embed_rec, self.rel_embeds, index_sample, label)
            ranks.append(rank)
            # losses.append(loss.item())
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)


class PostEnsembleBiDynamicRGCN(PostBiDynamicRGCN, PostEnsembleDynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(PostEnsembleBiDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, PostEnsembleEvaluationFilter)

    def init_freq_mlp(self):
        PostEnsembleDynamicRGCN.init_freq_mlp(self)

    def calc_ensemble_ratio(self, triples, t, g):
        return PostEnsembleDynamicRGCN.calc_ensemble_ratio(self, triples, t, g)

    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        return PostEnsembleDynamicRGCN.train_link_prediction(self, ent_embed, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True)

    def calc_metrics(self, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, g_list, t_list, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                     start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed_loc, ent_embed_rec in zip(g_list, t_list, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            time_diff_tensor_forward = cur_t - start_time_tensor_forward[i]
            time_diff_tensor_backward = cur_t - start_time_tensor_backward[i]
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_forward_loc[i], hist_embeddings_forward_rec[i][0], hist_embeddings_forward_rec[i][1],
                                                        time_diff_tensor_forward, hist_embeddings_backward_loc[i], hist_embeddings_backward_rec[i][0], hist_embeddings_backward_rec[i][1], time_diff_tensor_backward)

            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            # label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
            if index_sample.shape[0] == 0: continue
            weight_subject, weight_object = self.calc_ensemble_ratio(index_sample, t, g)
            rank = self.evaluater.calc_metrics_single_graph(ent_embed_loc, ent_embed_rec, self.rel_embeds, all_embeds_g_loc, all_embeds_g_rec, weight_subject, weight_object, index_sample, g, t)
            # loss = self.link_classification_loss(ent_embed, self.rel_embeds, index_sample, label)

            ranks.append(rank)
            # losses.append(loss.item())
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True)
        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False)

        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.train_seq_len, hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward,
                                                              hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full=False)

        i = 0
        for t, g, ent_embed_loc, ent_embed_rec in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            time_diff_tensor_forward = self.train_seq_len - 1 - start_time_tensor_forward[i]
            time_diff_tensor_backward = self.train_seq_len - 1 - start_time_tensor_backward[i]
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_forward_loc[i], hist_embeddings_forward_rec[i][0], hist_embeddings_forward_rec[i][1], time_diff_tensor_forward,
                                                                   hist_embeddings_backward_loc[i], hist_embeddings_backward_rec[i][0], hist_embeddings_backward_rec[i][1], time_diff_tensor_backward)
            weight_subject, weight_object = self.calc_ensemble_ratio(triplets, t, g)
            score_tail_local = self.train_link_prediction(ent_embed_loc, triplets, neg_tail_samples, labels, all_embeds_g_loc, corrupt_tail=True)
            score_head_local = self.train_link_prediction(ent_embed_loc, triplets, neg_head_samples, labels, all_embeds_g_loc, corrupt_tail=False)
            score_tail_temporal = self.train_link_prediction(ent_embed_rec, triplets, neg_tail_samples, labels, all_embeds_g_rec, corrupt_tail=True)
            score_head_temporal = self.train_link_prediction(ent_embed_rec, triplets, neg_head_samples, labels, all_embeds_g_rec, corrupt_tail=False)
            loss_tail = self.combined_scores(score_tail_local, score_tail_temporal, labels, weight_object)
            loss_head = self.combined_scores(score_head_local, score_head_temporal, labels, weight_subject)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(
            t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list, _, _ = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_forward_loc, hist_embeddings_forward_rec, start_time_tensor_forward = self.pre_forward(
            g_forward_batched_list, t_forward_batched_list, forward=True, val=True)
        hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward = self.pre_forward(
            g_backward_batched_list, t_backward_batched_list, forward=False, val=True)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.test_seq_len, hist_embeddings_forward_loc,
                    hist_embeddings_forward_rec, start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, full=True)

        return self.calc_metrics(per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings_forward_loc, hist_embeddings_forward_rec,
                                        start_time_tensor_forward, hist_embeddings_backward_loc, hist_embeddings_backward_rec, start_time_tensor_backward, self.test_seq_len - 1)
