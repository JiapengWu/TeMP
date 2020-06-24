import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.RGCN import RGCNLayer
import math
import pdb
from utils.utils import move_dgl_to_cuda, cuda

class SARGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=True, activation=None,
                 self_loop=True, dropout=0.0):
        super(SARGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases, total_times, bias, activation,
                                          self_loop, dropout)
        self.num_layers = args.num_layers
        self.q_linear = nn.Linear(in_feat, in_feat, bias=False)
        self.v_linear = nn.Linear(in_feat, in_feat, bias=False)
        self.k_linear = nn.Linear(in_feat, in_feat, bias=False)
        self.in_feat = in_feat
        self.h = 8
        self.d_k = in_feat // self.h
        self.post_aggregation = args.post_aggregation
        self.post_ensemble = args.post_ensemble

    def calc_result(self, cur_embeddings, prev_embeddings, time_diff, local_attn_mask):
        if self.learnable_lambda:
            decay_weight = -torch.clamp(self.exponential_decay(time_diff.unsqueeze(1)), min=0).squeeze()
        else:
            decay_weight = 0
        all_time_embeds = torch.cat([prev_embeddings, cur_embeddings.unsqueeze(1)], dim=1)
        bs = all_time_embeds.shape[0]
        q = self.q_linear(cur_embeddings).unsqueeze(1).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(all_time_embeds).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(all_time_embeds).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, local_attn_mask, decay_weight)
        return scores.transpose(1, 2).contiguous().view(bs, self.in_feat)

    def forward_final(self, g, prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes):
        # pdb.set_trace()
        current_graph, time_embedding = self.forward(g, time_batched_list_t, node_sizes)
        cur_embeddings = current_graph.ndata['h'] + time_embedding
        concat = self.calc_result(cur_embeddings, prev_embeddings, time_diff, local_attn_mask)

        if self.post_aggregation:
            return cur_embeddings, concat
        else:
            return current_graph, concat

    def attention(self, q, k, v, local_attn_mask, decay_weight):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        normalised = F.softmax(scores.squeeze() + local_attn_mask.unsqueeze(1) + decay_weight, dim=-1)
        output = torch.matmul(normalised.unsqueeze(2), v).squeeze()
        return output

    def forward_isolated(self, node_repr, prev_embeddings, time_diff, local_attn_mask, time):
        cur_embeddings, time_embedding = super().forward_isolated(node_repr, time)
        cur_time_embeddings = cur_embeddings + time_embedding
        concat = self.calc_result(cur_time_embeddings, prev_embeddings, time_diff, local_attn_mask)
        if self.post_aggregation:
            return cur_time_embeddings, concat
        else:
            return concat

    def forward_ema(self, g, prev_embeddings, time_batched_list_t, node_sizes, alpha, train_seq_len):
        current_graph, time_embedding = self.forward(g, time_batched_list_t, node_sizes)
        cur_embeddings = current_graph.ndata['h'] + time_embedding
        pdb.set_trace()
        all_time_embeds = torch.cat([prev_embeddings, cur_embeddings.unsqueeze(1)], dim=1)
        ema_vec = torch.pow(1 - alpha, cuda(torch.arange(train_seq_len)))
        ema_vec[:, :-1] *= alpha
        ema_vec = ema_vec.flip(-1).unsqueeze(0)
        averaged = torch.sum(all_time_embeds.transpose(1, 2) * ema_vec, -1)
        return averaged

    def forward_ema_isolated(self, node_repr, prev_embeddings, time, alpha, train_seq_len):
        cur_embeddings, time_embedding = super().forward_isolated(node_repr, time)
        # pdb.set_trace()
        all_time_embeds = torch.cat([prev_embeddings, (cur_embeddings + time_embedding).unsqueeze(1)], dim=1)
        ema_vec = torch.pow(1 - alpha, cuda(torch.arange(train_seq_len)))
        ema_vec[:, :-1] *= alpha
        ema_vec = ema_vec.flip(-1).unsqueeze(0)
        averaged = torch.sum(all_time_embeds.transpose(1, 2) * ema_vec, -1)
        return averaged


class SARGCN(nn.Module):
    # TODO: generalize to n layers
    def __init__(self, args, hidden_size, embed_size, num_rels, total_time):
        super(SARGCN, self).__init__()
        # in_feat = embed_size if static else hidden_size + embed_size
        self.rec_only_last_layer = args.rec_only_last_layer
        args.use_time_embedding = True
        if not self.rec_only_last_layer:
            self.layer_1 = SARGCNLayer(args, embed_size, hidden_size, 2 * num_rels, args.n_bases, total_time,
                                            activation=None, self_loop=True, dropout=args.dropout)
        else:
            self.layer_1 = RGCNLayer(args, embed_size, hidden_size, 2 * num_rels, args.n_bases, total_time,
                                           activation=None, self_loop=True, dropout=args.dropout)

        self.layer_2 = SARGCNLayer(args, hidden_size, hidden_size, 2 * num_rels, args.n_bases, total_time,
                                        activation=F.relu, self_loop=True, dropout=args.dropout)

    def forward(self, batched_graph, time_batched_list_t, node_sizes):
        # pdb.set_trace()
        first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)
        second_batched_graph, second_temp_embed = self.layer_2(first_batched_graph, time_batched_list_t, node_sizes)
        return first_batched_graph.ndata['h'] + first_temp_embed, second_batched_graph.ndata['h'] + second_temp_embed

    def forward_final(self, batched_graph, first_layer_prev_embeddings, second_layer_prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes):
        if not self.rec_only_last_layer:
            first_batched_graph, first_attn_embed = self.layer_1.forward_final(batched_graph, first_layer_prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes)
        else:
            first_batched_graph, _ = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_batched_graph, second_attn_embed = self.layer_2.forward_final(first_batched_graph, second_layer_prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes)
        # JK max pooling over the first and second layer
        return second_attn_embed if self.rec_only_last_layer else torch.max(torch.stack([first_attn_embed, second_attn_embed], dim=-1), dim=-1)[0]

    def forward_isolated(self, ent_embeds, first_layer_prev_embeddings, second_layer_prev_embeddings, time_diff, local_attn_mask, time):
        if not self.rec_only_last_layer:
            first_ent_embeds = self.layer_1.forward_isolated(ent_embeds, first_layer_prev_embeddings, time_diff, local_attn_mask, time)
        else:
            first_ent_embeds, _ = self.layer_1.forward_isolated(ent_embeds, time)
        second_ent_embeds = self.layer_2.forward_isolated(first_ent_embeds, second_layer_prev_embeddings, time_diff, local_attn_mask, time)
        return torch.max(torch.stack([first_ent_embeds, second_ent_embeds], dim=-1), dim=-1)[0] if not self.rec_only_last_layer else second_ent_embeds

    def forward_ema_isolated(self, ent_embeds, second_layer_prev_embeddings, time, alpha, train_seq_len):
        first_ent_embeds, _ = self.layer_1.forward_isolated(ent_embeds, time)
        second_ent_embeds = self.layer_2.forward_ema_isolated(first_ent_embeds, second_layer_prev_embeddings, time, alpha, train_seq_len)
        return second_ent_embeds

    def forward_ema(self, batched_graph, second_layer_prev_embeddings, time_batched_list_t, node_sizes, alpha, train_seq_len):
        first_batched_graph, _ = self.layer_1(batched_graph, time_batched_list_t, node_sizes)
        second_attn_embed = self.layer_2.forward_ema(first_batched_graph, second_layer_prev_embeddings, time_batched_list_t, node_sizes, alpha, train_seq_len)
        return second_attn_embed

    def forward_post_ensemble(self, batched_graph, second_layer_prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes):
        first_batched_graph, _ = self.layer_1(batched_graph, time_batched_list_t, node_sizes)
        second_local_embeds, second_attn_embed = self.layer_2.forward_final(first_batched_graph, second_layer_prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes)
        # JK max pooling over the first and second layer
        return second_local_embeds, second_attn_embed

    def forward_isolated_post_ensemble(self, ent_embeds, second_layer_prev_embeddings, time_diff, local_attn_mask, time):
        first_ent_embeds, _ = self.layer_1.forward_isolated(ent_embeds, time)
        second_local_embeds, second_ent_embeds = self.layer_2.forward_isolated(first_ent_embeds, second_layer_prev_embeddings, time_diff, local_attn_mask, time)
        return second_local_embeds, second_ent_embeds
