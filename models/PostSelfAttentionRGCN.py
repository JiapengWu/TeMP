from models.SelfAttentionRGCN import SelfAttentionRGCN
from models.BiSelfAttentionRGCN import BiSelfAttentionRGCN
from utils.post_evaluation import PostEvaluationFilter
from models.PostBiDynamicRGCN import PostBiDynamicRGCN
from models.PostDynamicRGCN import PostDynamicRGCN
from utils.utils import move_dgl_to_cuda, cuda, filter_none
from models.BiDynamicRGCN import BiDynamicRGCN
import torch
from utils.DropEdge import DropEdge
import numpy as np
import time


class PostSelfAttentionRGCN(SelfAttentionRGCN, PostDynamicRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test,
                 evaluater_type=PostEvaluationFilter):
        super(PostSelfAttentionRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val,
                                                      graph_dict_test, evaluater_type)
        # start = time.time()
        self.drop_edge = DropEdge(args, graph_dict_train, graph_dict_val, graph_dict_test)
        # print("After {} ,done".format(time.time() - start))
        self.init_freq_mlp()

    def get_all_embeds_Gt(self, convoluted_embeds_loc, convoluted_embeds_rec, g, t, first_prev_graph_embeds, second_prev_graph_embeds, attn_mask, val=False):
        all_embeds_g_loc = self.ent_embeds.new_zeros(self.ent_embeds.shape)
        all_embeds_g_rec = self.ent_embeds.new_zeros(self.ent_embeds.shape)

        res_all_embeds_g_loc, res_all_embeds_g_rec = self.ent_encoder.forward_isolated_post_ensemble(self.ent_embeds, second_prev_graph_embeds.transpose(0, 1),
                                                         self.time_diff_test if val else self.time_diff_train, attn_mask.transpose(0, 1), t)
        all_embeds_g_loc[:] = res_all_embeds_g_loc[:]
        all_embeds_g_rec[:] = res_all_embeds_g_rec[:]

        keys = np.array(list(g.ids.keys()))
        values = np.array(list(g.ids.values()))
        # pdb.set_trace()
        all_embeds_g_loc[values] = convoluted_embeds_loc[keys]
        all_embeds_g_rec[values] = convoluted_embeds_rec[keys]

        return all_embeds_g_loc, all_embeds_g_rec

    def get_final_graph_embeds(self, g_batched_list_t, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full, rate=0.5, val=False):
        batched_graph = self.get_batch_graph_embeds(g_batched_list_t, full, rate)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        first_layer_prev_embeddings, second_layer_prev_embeddings, local_attn_mask = self.get_prev_embeddings(g_batched_list_t, hist_embeddings, attn_mask)
        second_local_embeds, second_layer_embeds = self.ent_encoder.forward_post_ensemble(batched_graph, second_layer_prev_embeddings,
                                                             self.time_diff_test if val else self.time_diff_train, local_attn_mask, time_batched_list_t, node_sizes)
        return second_local_embeds.split(node_sizes), second_layer_embeds.split(node_sizes)

    def pre_forward(self, g_batched_list, time_batched_list, val=False):
        seq_len = self.test_seq_len if val else self.train_seq_len
        bsz = len(g_batched_list[0])
        target_time_batched_list = time_batched_list[-1]
        hist_embeddings = self.ent_embeds.new_zeros(seq_len - 1, bsz, 2, self.num_ents, self.embed_size)
        attn_mask = self.ent_embeds.new_zeros(seq_len, bsz, self.num_ents) - 10e9
        attn_mask[-1] = 0
        full = val or not self.args.random_dropout
        for cur_t in range(seq_len - 1):
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, cur_t)
            if len(g_batched_list_t) == 0: continue
            first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(g_batched_list_t, time_batched_list[cur_t], node_sizes, full=full, rate=0.8)
            self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, hist_embeddings, g_batched_list_t, cur_t, attn_mask, bsz)
        return hist_embeddings, attn_mask

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_batched_list, time_batched_list = self.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)
        hist_embeddings, attn_mask = self.pre_forward(g_batched_list, time_batched_list, val=False)

        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full=False, val=False)

        i = 0
        for t, g, ent_embed_loc, ent_embed_rec in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec ):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, \
                                                            weight_object_query_object_embed = self.calc_ensemble_ratio(triplets, t, g)
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings[:, i, 0], hist_embeddings[:, i, 1], attn_mask[:, i], val=False)

            loss_tail = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_tail_samples, labels, all_embeds_g_loc,
                                                   all_embeds_g_rec, weight_object_query_subject_embed, weight_object_query_object_embed, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_head_samples, labels, all_embeds_g_loc,
                                                   all_embeds_g_rec, weight_subject_query_subject_embed, weight_subject_query_object_embed, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            i += 1
        return reconstruct_loss

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_train_batched_list, time_list = self.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, _ = self.get_batch_graph_list(t_list, 1, graph_dict)
        hist_embeddings, attn_mask = self.pre_forward(g_train_batched_list, time_list, val=True)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs = g_train_batched_list[-1]

        node_sizes = [len(g.nodes()) for g in train_graphs]
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_list[-1], node_sizes, hist_embeddings, attn_mask, full=True, val=True)
        return self.calc_metrics(per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_list[-1], hist_embeddings, attn_mask)

    def calc_metrics(self, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, g_list, t_list, hist_embeddings, attn_mask):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        i = 0
        for g, t, ent_embed_loc, ent_embed_rec in zip(g_list, t_list, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings[:, i, 0], hist_embeddings[:, i, 1], attn_mask[:, i], val=True)
            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                label = cuda(label)
            if index_sample.shape[0] == 0: continue
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed = self.calc_ensemble_ratio(index_sample, t, g)
            rank = self.evaluater.calc_metrics_single_graph(ent_embed_loc, ent_embed_rec, self.rel_embeds, all_embeds_g_loc, all_embeds_g_rec, index_sample, weight_subject_query_subject_embed,
                                                            weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed, g, t)
            ranks.append(rank)
            i += 1
        try:
            ranks = torch.cat(ranks)
        except:
            ranks = cuda(torch.tensor([]).long()) if self.use_cuda else torch.tensor([]).long()

        return ranks, np.mean(losses)


class PostBiSelfAttentionRGCN(BiSelfAttentionRGCN, PostSelfAttentionRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=PostEvaluationFilter):
        super(PostBiSelfAttentionRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)
        # print("Calculating frequencies...")
        start = time.time()
        self.drop_edge = DropEdge(args, graph_dict_train, graph_dict_val, graph_dict_test)
        # print("After {} ,done".format(time.time() - start))
        self.init_freq_mlp()

    def pre_forward(self, g_batched_list, time_batched_list, forward=True, val=False):
        seq_len = self.test_seq_len if val else self.train_seq_len
        bsz = len(g_batched_list[0])
        hist_embeddings = self.ent_embeds.new_zeros(seq_len - 1, bsz, 2, self.num_ents, self.embed_size)
        attn_mask = self.ent_embeds.new_zeros(seq_len - 1, bsz, self.num_ents) - 10e9
        target_time_batched_list = time_batched_list[-1]
        for cur_t in range(seq_len - 1):
            g_batched_list_t, node_sizes = self.get_val_vars(g_batched_list, cur_t)
            if len(g_batched_list_t) == 0: continue
            if self.edge_dropout and not val:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_dropout_embeds(
                    time_batched_list[cur_t], target_time_batched_list, node_sizes)
            else:
                first_per_graph_ent_embeds, second_per_graph_ent_embeds = self.get_per_graph_ent_embeds(
                    g_batched_list_t, time_batched_list[cur_t], node_sizes, full=True, rate=0.8)

            self.update_time_diff_hist_embeddings(first_per_graph_ent_embeds, second_per_graph_ent_embeds, hist_embeddings, g_batched_list_t, cur_t, attn_mask, bsz)
        if not forward:
            hist_embeddings = torch.flip(hist_embeddings, [1])
            attn_mask = torch.flip(attn_mask, [1])
        return hist_embeddings, attn_mask

    def forward(self, t_list, reverse=False):
        reconstruct_loss = 0
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = BiDynamicRGCN.get_batch_graph_list(t_list, self.train_seq_len, self.graph_dict_train)

        hist_embeddings_forward, attn_mask_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True)
        hist_embeddings_backward, attn_mask_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False)
        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]

        node_sizes = [len(g.nodes()) for g in train_graphs]
        hist_embeddings = torch.cat([hist_embeddings_forward, hist_embeddings_backward], dim=0)  # 2 * seq_len - 2, bsz, num_ents
        attn_mask = torch.cat([attn_mask_forward, attn_mask_backward, attn_mask_forward.new_zeros(1, *attn_mask_forward.shape[1:])], dim=0) # 2 * seq_len - 1, bsz, num_ents
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full=False)

        i = 0
        for t, g, ent_embed_loc, ent_embed_rec in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds_loc, per_graph_ent_embeds_rec):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            # import pdb; pdb.set_trace()
            all_embeds_g_loc, all_embeds_g_rec = self.get_all_embeds_Gt(ent_embed_loc, ent_embed_rec, g, t, hist_embeddings[:, i, 0], hist_embeddings[:, i, 1], attn_mask[:, i], val=False)
            weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed = self.calc_ensemble_ratio(triplets, t, g)
            loss_tail = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_tail_samples, labels, all_embeds_g_loc,
                                                   all_embeds_g_rec, weight_object_query_subject_embed, weight_object_query_object_embed, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed_loc, ent_embed_rec, triplets, neg_head_samples, labels, all_embeds_g_loc,
                                                   all_embeds_g_rec, weight_subject_query_subject_embed, weight_subject_query_object_embed, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
            del all_embeds_g_loc, all_embeds_g_rec, weight_subject_query_subject_embed, weight_subject_query_object_embed, weight_object_query_subject_embed, weight_object_query_object_embed, triplets, neg_tail_samples, neg_head_samples, labels
            # pdb.set_trace()
            i += 1
        return reconstruct_loss

    def evaluate(self, t_list, val=True):
        graph_dict = self.graph_dict_val if val else self.graph_dict_test
        g_forward_batched_list, t_forward_batched_list, g_backward_batched_list, t_backward_batched_list = BiDynamicRGCN.get_batch_graph_list(t_list, self.test_seq_len, self.graph_dict_train)
        g_val_batched_list, val_time_list, _, _ = BiDynamicRGCN.get_batch_graph_list(t_list, 1, graph_dict)

        hist_embeddings_forward, attn_mask_forward = self.pre_forward(g_forward_batched_list, t_forward_batched_list, forward=True)
        hist_embeddings_backward, attn_mask_backward = self.pre_forward(g_backward_batched_list, t_backward_batched_list, forward=False)

        test_graphs, _ = self.get_val_vars(g_val_batched_list, -1)
        train_graphs, time_batched_list_t = g_forward_batched_list[-1], t_forward_batched_list[-1]

        node_sizes = [len(g.nodes()) for g in train_graphs]
        hist_embeddings = torch.cat([hist_embeddings_forward, hist_embeddings_backward], dim=0)  # 2 * seq_len - 2, bsz, num_ents
        attn_mask = torch.cat([attn_mask_forward, attn_mask_backward, attn_mask_forward.new_zeros(1, *attn_mask_forward.shape[1:])], dim=0) # 2 * seq_len - 1, bsz, num_ents
        per_graph_ent_embeds_loc, per_graph_ent_embeds_rec = self.get_final_graph_embeds(train_graphs, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full=True)

        return self.calc_metrics(per_graph_ent_embeds_loc, per_graph_ent_embeds_rec, test_graphs, time_batched_list_t, hist_embeddings, attn_mask)
