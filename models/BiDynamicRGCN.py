from models.BiRRGCN import BiRRGCN
import numpy as np
from utils.utils import move_dgl_to_cuda, cuda, filter_none
from utils.scores import *
from models.DynamicRGCN import DynamicRGCN
import pdb
from utils.evaluation import EvaluationFilter


class BiDynamicRGCN(DynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=EvaluationFilter):
        super(BiDynamicRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)

    def build_model(self):
        self.ent_encoder = BiRRGCN(self.args, self.hidden_size, self.embed_size, self.num_rels, self.total_time)

    @staticmethod
    def get_batch_graph_list(t_list, seq_len, graph_dict):
        times = list(graph_dict.keys())
        # time_unit = times[1] - times[0]  # compute time unit
        t_list_forward = t_list.sort(descending=True)[0]
        t_list_backward = t_list.sort(descending=False)[0]
        time_list_forward = []
        g_list_forward = []

        time_list_backward = []
        g_list_backward= []
        num_times = len(times)

        for tim in t_list_forward:
            length_forward = times.index(tim) + 1
            time_seq_forward = times[length_forward - seq_len:length_forward] if seq_len <= length_forward else times[:length_forward]
            time_list_forward.append(([None] * (seq_len - len(time_seq_forward))) + time_seq_forward)
            g_list_forward.append(([None] * (seq_len - len(time_seq_forward))) + [graph_dict[t] for t in time_seq_forward])

        for tim in t_list_backward:
            length_backward = times.index(tim)
            time_seq_backward = times[length_backward:length_backward + seq_len] if seq_len <= num_times - length_backward else times[length_backward:]
            time_seq_backward.reverse()
            time_list_backward.append(([None] * (seq_len - len(time_seq_backward))) + time_seq_backward)
            g_list_backward.append(([None] * (seq_len - len(time_seq_backward))) + [graph_dict[t] for t in time_seq_backward])

        t_forward_batched_list = [list(x) for x in zip(*time_list_forward)]
        g_forward_batched_list = [list(x) for x in zip(*g_list_forward)]

        t_backward_batched_list = [list(x) for x in zip(*time_list_backward)]
        g_backward_batched_list = [list(x) for x in zip(*g_list_backward)]

        return g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list

    def get_per_graph_ent_dropout_embeds_one_direction(self, cur_time_list, target_time_list, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward):
        batched_graph = self.get_batch_graph_dropout_embeds(filter_none(cur_time_list), target_time_list)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_embeds, second_layer_embeds = self.ent_encoder.forward_one_direction(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, cur_time_list, node_sizes, forward)

        return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_per_graph_ent_embeds_one_direction(self, g_batched_list_t, time_batched_list_t, node_sizes, time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward, full, rate=0.5):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full=full, rate=rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_embeds, second_layer_embeds = self.ent_encoder.forward_one_direction(batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward)

        return first_layer_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def get_graph_embeds_center(self, g_batched_list_t, time_batched_list_t, node_sizes, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                               first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, full, rate=0.5):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        second_layer_embeds = self.ent_encoder(batched_graph, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                                 first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes)

        return second_layer_embeds.split(node_sizes)

    def pre_forward(self, g_batched_list, time_batched_list, forward=True, val=False):
        seq_len = self.test_seq_len if val else self.train_seq_len
        bsz = len(g_batched_list[0])
        target_time_batched_list = time_batched_list[-1]
        hist_embeddings = self.ent_embeds.new_zeros(bsz, 2, self.num_ents, self.embed_size)
        start_time_tensor = self.ent_embeds.new_zeros(bsz, self.num_ents)
        full = val or not self.args.random_dropout
        for cur_t in range(seq_len - 1):
            # if not forward: pdb.set_trace()
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, cur_t)

            if len(g_batched_list_t) == 0: continue
            first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor = self.get_prev_embeddings(g_batched_list_t, hist_embeddings, start_time_tensor, cur_t)
            if self.edge_dropout and not val:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_dropout_embeds_one_direction(time_batched_list[cur_t], target_time_batched_list, node_sizes,
                                                                                                                  time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward)
            else:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds_one_direction(g_batched_list_t, time_batched_list[cur_t], node_sizes,
                                                                                                                  time_diff_tensor, first_prev_graph_embeds, second_prev_graph_embeds, forward, full=full, rate=0.8)
            hist_embeddings = self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, start_time_tensor, g_batched_list_t, cur_t, bsz)
        if not forward:
            hist_embeddings = torch.flip(hist_embeddings, [0])
            start_time_tensor = torch.flip(start_time_tensor, [0])
        return hist_embeddings, start_time_tensor

    def get_all_embeds_Gt(self, convoluted_embeds, g, t, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                         first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward):
        all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)
        if self.args.use_embed_for_non_active:
            all_embeds_g[:] = self.ent_embeds[:]
        else:
            all_embeds_g = self.ent_encoder.forward_isolated(self.ent_embeds, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward.unsqueeze(-1),
                                                                              first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward.unsqueeze(-1), t)
        for k, v in g.ids.items():
            all_embeds_g[v] = convoluted_embeds[k]
        return all_embeds_g

    def get_final_graph_embeds(self, g_batched_list_t, time_batched_list_t, seq_len, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward, full):
        first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward = self.get_prev_embeddings(g_batched_list_t, hist_embeddings_forward, start_time_tensor_forward, seq_len - 1)
        first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward = self.get_prev_embeddings(g_batched_list_t, hist_embeddings_backward, start_time_tensor_backward, seq_len - 1)

        node_sizes = [len(g.nodes()) for g in g_batched_list_t]

        return self.get_graph_embeds_center(g_batched_list_t, time_batched_list_t, node_sizes, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                            first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, full=full)

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings_forward, start_time_tensor_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True)
        hist_embeddings_backward, start_time_tensor_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False)

        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.train_seq_len, hist_embeddings_forward, start_time_tensor_forward,
                                                              hist_embeddings_backward, start_time_tensor_backward, full=False)

        i = 0
        for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            time_diff_tensor_forward = self.train_seq_len - 1 - start_time_tensor_forward[i]
            time_diff_tensor_backward = self.train_seq_len - 1 - start_time_tensor_backward[i]
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_forward[i][0], hist_embeddings_forward[i][1], time_diff_tensor_forward,
                                                                   hist_embeddings_backward[i][0], hist_embeddings_backward[i][1], time_diff_tensor_backward)
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate(self, t_list, val=True):
        per_graph_ent_embeds, test_graphs, time_batched_list_t, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward = self.evaluate_embed(t_list, val)
        return self.calc_metrics(per_graph_ent_embeds, test_graphs, time_batched_list_t,
                                 hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward, self.test_seq_len - 1)

    def evaluate_embed(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list, _, _ = self.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_forward, start_time_tensor_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True, val=True)
        hist_embeddings_backward, start_time_tensor_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False, val=True)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.test_seq_len, hist_embeddings_forward, start_time_tensor_forward,
                                                              hist_embeddings_backward, start_time_tensor_backward, full=True)
        return per_graph_ent_embeds, test_graphs, time_batched_list_t, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward

    def train_embed(self, t_list):
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = self.get_batch_graph_list(
            t_list, self.test_seq_len, self.graph_dict_train)

        hist_embeddings_forward, start_time_tensor_forward = self.pre_forward(g_forward_batched_list,
                                                                              t_forward_batched_list, forward=True,
                                                                              val=True)
        hist_embeddings_backward, start_time_tensor_backward = self.pre_forward(g_backward_batched_list,
                                                                                t_backward_batched_list, forward=False,
                                                                                val=True)

        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_batched_list_t, self.test_seq_len,
                                                           hist_embeddings_forward, start_time_tensor_forward,
                                                           hist_embeddings_backward, start_time_tensor_backward,
                                                           full=True)
        return per_graph_ent_embeds, train_graphs, time_batched_list_t, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward

    def calc_metrics(self, per_graph_ent_embeds, g_list, t_list, hist_embeddings_forward, start_time_tensor_forward, hist_embeddings_backward, start_time_tensor_backward, cur_t):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed in zip(g_list, t_list, per_graph_ent_embeds):
            time_diff_tensor_forward = cur_t - start_time_tensor_forward[i]
            time_diff_tensor_backward = cur_t - start_time_tensor_backward[i]
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_forward[i][0], hist_embeddings_forward[i][1], time_diff_tensor_forward,
                                                  hist_embeddings_backward[i][0], hist_embeddings_backward[i][1], time_diff_tensor_backward)

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