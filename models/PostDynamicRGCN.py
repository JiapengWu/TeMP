from models.DynamicRGCN import DynamicRGCN
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
from utils.evaluation import EvaluationFilter
from utils.post_evaluation import PostEvaluationFilter, PostEnsembleEvaluationFilter
import pdb
from collections import defaultdict
# from utils.frequency import count_freq_per_time, calc_aggregated_statistics
from utils.dataset import load_quadruples

class ImputeDynamicRGCN(DynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=EvaluationFilter):
        super(ImputeDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)

    def get_all_embeds_Gt(self, convoluted_embeds, g, t, second_embeds_loc, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor):
        # all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)
        all_embeds_g = self.ent_encoder.forward_isolated_impute(self.ent_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor.unsqueeze(-1), t, second_embeds_loc)

        keys = np.array(list(g.ids.keys()))
        values = np.array(list(g.ids.values()))
        all_embeds_g[values] = convoluted_embeds[keys]
        return all_embeds_g

    def update_time_diff_hist_embeddings(self, second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, cur_t, bsz):
        loc = start_time_tensor.new_zeros(bsz, self.num_ents, self.embed_size)
        rec = start_time_tensor.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        for i in range(len(first_per_graph_ent_embeds)):
            idx = g_batched_list_t[i].ndata['id'].squeeze()
            loc[i][idx] = second_local_embeds[i]
            rec[i][0][idx] = first_per_graph_ent_embeds[i]
            rec[i][1][idx] = second_per_graph_ent_embeds[i]
            start_time_tensor[i][idx] = cur_t
        return loc, rec

    def get_per_graph_ent_embeds(self, g_batched_list_t, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full, rate=0.5):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        second_local_embeds, first_layer_embeds, second_layer_embeds = self.ent_encoder.forward_post_ensemble(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes)

        return second_local_embeds.split(node_sizes), first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_per_graph_ent_dropout_embeds(self, cur_time_list, target_time_list, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds):
        batched_graph = self.get_batch_graph_dropout_embeds(filter_none(cur_time_list), target_time_list)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        second_local_embeds, first_layer_embeds, second_layer_embeds = self.ent_encoder.forward_post_ensemble(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, cur_time_list, node_sizes)

        return second_local_embeds.split(node_sizes), first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def pre_forward(self, g_batched_list, time_batched_list, val=False):
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
                second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_dropout_embeds(time_batched_list[cur_t], target_time_batched_list, node_sizes,
                                                                                                        time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds)
            else:
                second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, time_batched_list[cur_t], node_sizes,
                                                                                                        time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=full, rate=0.8)
            hist_embeddings_loc, hist_embeddings_rec = self.update_time_diff_hist_embeddings(second_local_embeds, first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, cur_t, bsz)
        return hist_embeddings_loc, hist_embeddings_rec, start_time_tensor

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.pre_forward(g_batched_list, time_batched_list)

        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(train_graphs, hist_embeddings_rec, start_time_tensor, self.train_seq_len - 1)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        _, _, per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graphs, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=False)

        i = 0
        for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_loc[i], hist_embeddings_rec[i][0], hist_embeddings_rec[i][1], self.train_seq_len - 1 - start_time_tensor[i])
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate_embed(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.pre_forward(g_train_batched_list, time_list, val=True)
        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs = g_train_batched_list[-1]

        first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(train_graphs, hist_embeddings_rec, start_time_tensor, self.test_seq_len - 1)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        _, _, per_graph_ent_embeds = self.get_per_graph_ent_embeds(train_graphs, time_list[-1], node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=True)
        return per_graph_ent_embeds, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor

    def evaluate(self, t_list, val=True):
        per_graph_ent_embeds, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds, test_graphs, time_list[-1], hist_embeddings_loc, hist_embeddings_rec, start_time_tensor, self.test_seq_len - 1)

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):
            time_diff_tensor = cur_t - start_time_tensor[i]
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_loc[i], hist_embeddings_rec[i][0], hist_embeddings_rec[i][1], time_diff_tensor)

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


class PostDynamicRGCN(ImputeDynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=PostEvaluationFilter):
        super(PostDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)
        self.drop_edge = DropEdge(args, graph_dict_train, graph_dict_val, graph_dict_test)
        self.init_freq_mlp()

    def init_freq_mlp(self):
        self.subject_query_subject_embed_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.object_query_subject_embed_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.subject_query_object_embed_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.object_query_object_embed_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def get_all_embeds_Gt(self, convoluted_embeds_loc, convoluted_embeds, g, t, second_embeds_loc, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor):
        # all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)
        all_embeds_g_loc, all_embeds_g_rec = self.ent_encoder.forward_post_ensemble_isolated(self.ent_embeds, first_prev_graph_embeds,
                                                                second_prev_graph_embeds, time_diff_tensor.unsqueeze(-1), t, second_embeds_loc)

        keys = np.array(list(g.ids.keys()))
        values = np.array(list(g.ids.values()))
        all_embeds_g_loc[values] = convoluted_embeds_loc[keys]
        all_embeds_g_rec[values] = convoluted_embeds[keys]

        # for k, v in g.ids.items():
        #     all_embeds_g_loc[v] = convoluted_embeds_loc[k]
        #     all_embeds_g_rec[v] = convoluted_embeds[k]
        return all_embeds_g_loc, all_embeds_g_rec

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.pre_forward(g_batched_list, time_batched_list)

        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(train_graphs, hist_embeddings_rec, start_time_tensor, self.train_seq_len - 1)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds_loc, _, per_graph_ent_embeds_rec = self.get_per_graph_ent_embeds(train_graphs, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=False)

        i = 0
        for t, g, ent_embed_loc, ent_embed_rec in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_loc[i], hist_embeddings_rec[i][0], hist_embeddings_rec[i][1], self.train_seq_len - 1 - start_time_tensor[i])
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed = self.calc_ensemble_ratio(triplets, t, g)
            loss_tail = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_tail_samples, labels, all_embeds_g_loc, all_embeds_g_rec, weight_object_query_subject_embed, weight_object_query_object_embed, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_head_samples, labels, all_embeds_g_loc, all_embeds_g_rec, weight_subject_query_subject_embed, weight_subject_query_object_embed, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate_embed(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.pre_forward(g_train_batched_list, time_list, val=True)
        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs = g_train_batched_list[-1]

        first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(train_graphs, hist_embeddings_rec, start_time_tensor, self.test_seq_len - 1)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds_loc, _, per_graph_ent_embeds_rec = self.get_per_graph_ent_embeds(train_graphs, time_list[-1], node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=True)
        return per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor

    def evaluate(self, t_list, val=True):
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list[-1], hist_embeddings_loc, hist_embeddings_rec, start_time_tensor, self.test_seq_len - 1)

    @staticmethod
    def combined_embeddings(embed_loc, embed_rec, weight):
        return weight * embed_loc + (1 - weight) * embed_rec

    def calc_metrics(self, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, g_list, t_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed_loc, ent_embed_rec in zip(g_list, t_list, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            time_diff_tensor = cur_t - start_time_tensor[i]

            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_loc[i], hist_embeddings_rec[i][0], hist_embeddings_rec[i][1], time_diff_tensor)
            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                label = cuda(label)
            if index_sample.shape[0] == 0: continue
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed = self.calc_ensemble_ratio(index_sample, t, g)
            # pdb.set_trace()
            rank = self.evaluater.calc_metrics_single_graph(ent_embed_loc, ent_embed_rec, self.rel_embeds, all_embeds_g_loc, all_embeds_g_rec,
                                                            index_sample, weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed, g, t)

            ranks.append(rank)
            # losses.append(loss.item())
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)

    def train_link_prediction(self, ent_embed_loc, ent_embed_rec, triplets, neg_samples, labels, all_embeds_g_loc, all_embeds_g_rec, weight_subject, weight_object, corrupt_tail=True):
        r = self.rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s_loc = ent_embed_loc[triplets[:, 0]]
            s_rec = ent_embed_rec[triplets[:, 0]]
            neg_o_loc = all_embeds_g_loc[neg_samples]
            neg_o_rec = all_embeds_g_rec[neg_samples]
            s = weight_subject * s_loc + (1 - weight_subject) * s_rec
            neg_o = weight_object.unsqueeze(-1) * neg_o_loc + (1 - weight_object).unsqueeze(-1) * neg_o_rec
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s_loc = all_embeds_g_loc[neg_samples]
            neg_s_rec = all_embeds_g_rec[neg_samples]
            o_loc = ent_embed_rec[triplets[:, 2]]
            o_rec = ent_embed_rec[triplets[:, 2]]
            neg_s = weight_subject.unsqueeze(-1) * neg_s_loc + (1 - weight_subject).unsqueeze(-1) * neg_s_rec
            o = weight_object * o_loc + (1 - weight_object) * o_rec
            score = self.calc_score(neg_s, r, o, mode='head')

        predict_loss = F.cross_entropy(score, labels)
        del score, labels
        return predict_loss

    def calc_ensemble_ratio(self, triples, t, g):
        sub_feature_vecs = []
        obj_feature_vecs = []
        t = t.item()
        for s, r, o in triples:

            s = g.ids[s.item()]
            r = r.item()
            o = g.ids[o.item()]
            # triple_freq = self.drop_edge.triple_freq_per_time_step_agg[t][(s, r, o)]
            # ent_pair_freq = self.drop_edge.ent_pair_freq_per_time_step_agg[t][(s, o)]
            sub_freq = self.drop_edge.sub_freq_per_time_step_agg[t][s]
            obj_freq = self.drop_edge.obj_freq_per_time_step_agg[t][o]
            rel_freq = self.drop_edge.rel_freq_per_time_step_agg[t][r]
            sub_rel_freq = self.drop_edge.sub_rel_freq_per_time_step_agg[t][(s, r)]
            obj_rel_freq = self.drop_edge.obj_rel_freq_per_time_step_agg[t][(o, r)]
            # 0: no local, 1: no temporal

            sub_feature_vecs.append(torch.tensor([obj_freq, rel_freq, obj_rel_freq]))
            obj_feature_vecs.append(torch.tensor([sub_freq, rel_freq, sub_rel_freq]))

        try:
            sub_features = torch.stack(sub_feature_vecs).float()
            obj_features = torch.stack(obj_feature_vecs).float()
            if self.use_cuda:
                sub_features = cuda(sub_features)
                obj_features = cuda(obj_features)
            weight_subject_query_subject_embed = torch.sigmoid(self.subject_query_subject_embed_linear(sub_features))
            weight_subject_query_object_embed = torch.sigmoid(self.subject_query_subject_embed_linear(sub_features))
            weight_object_query_subject_embed = torch.sigmoid(self.object_query_subject_embed_linear(obj_features))
            weight_object_query_object_embed = torch.sigmoid(self.object_query_subject_embed_linear(obj_features))
        except:
            weight_subject_query_subject_embed = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()
            weight_subject_query_object_embed = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()
            weight_object_query_subject_embed = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()
            weight_object_query_object_embed = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed


class PostEnsembleDynamicRGCN(PostDynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(PostEnsembleDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, PostEnsembleEvaluationFilter)

    def init_freq_mlp(self):
        self.subject_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.object_linear = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.pre_forward(g_batched_list, time_batched_list)

        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(train_graphs, hist_embeddings_rec, start_time_tensor, self.train_seq_len - 1)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds_loc, _, per_graph_ent_embeds_rec = self.get_per_graph_ent_embeds(train_graphs, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=False)

        i = 0
        for t, g, ent_embed_loc, ent_embed_rec in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_loc[i], hist_embeddings_rec[i][0], hist_embeddings_rec[i][1], self.train_seq_len - 1 - start_time_tensor[i])

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
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list[-1], hist_embeddings_loc, hist_embeddings_rec, start_time_tensor, self.test_seq_len - 1)

    def evaluate_embed(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_loc, hist_embeddings_rec, start_time_tensor = self.pre_forward(g_train_batched_list, time_list, val=True)
        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs = g_train_batched_list[-1]

        first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(train_graphs, hist_embeddings_rec, start_time_tensor, self.test_seq_len - 1)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds_loc, _, per_graph_ent_embeds_rec = self.get_per_graph_ent_embeds(train_graphs, time_list[-1], node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, full=True)
        return per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor

    def calc_metrics(self, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, g_list, t_list, hist_embeddings_loc, hist_embeddings_rec, start_time_tensor, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed_loc, ent_embed_rec in zip(g_list, t_list, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            time_diff_tensor = cur_t - start_time_tensor[i]

            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings_loc[i], hist_embeddings_rec[i][0], hist_embeddings_rec[i][1], time_diff_tensor)
            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                label = cuda(label)
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

    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        r = self.rel_embeds[triplets[:, 1]]
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = all_embeds_g[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = all_embeds_g[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')
        return score

    def combined_scores(self, local_score, temporal_score, labels, weight):
        score = weight * local_score + (1 - weight) * temporal_score
        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def calc_ensemble_ratio(self, triples, t, g):
        sub_feature_vecs = []
        obj_feature_vecs = []
        t = t.item()
        for s, r, o in triples:

            s = g.ids[s.item()]
            r = r.item()
            o = g.ids[o.item()]
            # triple_freq = self.drop_edge.triple_freq_per_time_step_agg[t][(s, r, o)]
            # ent_pair_freq = self.drop_edge.ent_pair_freq_per_time_step_agg[t][(s, o)]
            sub_freq = self.drop_edge.sub_freq_per_time_step_agg[t][s]
            obj_freq = self.drop_edge.obj_freq_per_time_step_agg[t][o]
            rel_freq = self.drop_edge.rel_freq_per_time_step_agg[t][r]
            sub_rel_freq = self.drop_edge.sub_rel_freq_per_time_step_agg[t][(s, r)]
            obj_rel_freq = self.drop_edge.obj_rel_freq_per_time_step_agg[t][(o, r)]
            # 0: no local, 1: no temporal
            sub_feature_vecs.append(torch.tensor([obj_freq, rel_freq, obj_rel_freq]))
            obj_feature_vecs.append(torch.tensor([sub_freq, rel_freq, sub_rel_freq]))

        try:
            sub_features = torch.stack(sub_feature_vecs).float()
            obj_features = torch.stack(obj_feature_vecs).float()
            if self.use_cuda:
                sub_features = cuda(sub_features)
                obj_features = cuda(obj_features)
            weight_subject = torch.sigmoid(self.subject_linear(sub_features))
            weight_object = torch.sigmoid(self.object_linear(obj_features))
        except:
            weight_subject = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()
            weight_object = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return weight_subject, weight_object