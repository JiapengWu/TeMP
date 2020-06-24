from torch import nn
from utils.scores import *
from utils.utils import cuda
import pdb
from models.SelfAttentionRGCN import SelfAttentionRGCN
from models.BiDynamicRGCN import BiDynamicRGCN
from utils.evaluation import EvaluationFilter
import numpy as np

class BiSelfAttentionRGCN(SelfAttentionRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type=EvaluationFilter):
        super(BiSelfAttentionRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test, evaluater_type)
        self.EMA = self.args.EMA
        if self.EMA:
            self.alpha = nn.Parameter(torch.Tensor(self.embed_size, 1))

    def build_model(self):
        super().build_model()
        self.time_diff_test = torch.tensor(list(range(self.test_seq_len - 1, 0, -1)) + list(range(self.test_seq_len - 1, 0, -1)) + [0.])
        self.time_diff_train = torch.tensor(list(range(self.train_seq_len - 1, 0, -1)) + list(range(self.train_seq_len - 1, 0, -1)) + [0.])
        if self.use_cuda:
            self.time_diff_test = cuda(self.time_diff_test)
            self.time_diff_train = cuda(self.time_diff_train)

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
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full=False)

        i = 0
        for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds):
            triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
            all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings[:, i, 0], hist_embeddings[:, i, 1], attn_mask[:, i])
            loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
            loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
            reconstruct_loss += loss_tail + loss_head
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
        per_graph_ent_embeds = self.get_final_graph_embeds(train_graphs, time_batched_list_t, node_sizes, hist_embeddings, attn_mask, full=True)

        return self.calc_metrics(per_graph_ent_embeds, test_graphs, time_batched_list_t, hist_embeddings, attn_mask)
