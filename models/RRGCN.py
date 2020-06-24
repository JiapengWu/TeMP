import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RGCN import RGCNLayer
from models.GRU_cell import GRUCell
'''
class DRGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super(DRGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases,
                                         total_times, bias, activation, self_loop, dropout)
        self.inv_temperature = args.inv_temperature

        self.time_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.time_weight,
                                gain=nn.init.calculate_gain('relu'))

        # self.time_projection = nn.Linear(in_feat, out_feat)

    def forward(self, g, prev_graph_embeds, time_diff_tensor):
        # for g in g_list:
        g = g.local_var()
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)
        # apply bias and activation
        node_repr = g.ndata['h']

        # node_repr = node_repr + self.time_projection(prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature))
        node_repr = node_repr + torch.mm(prev_graph_embeds, self.time_weight) * torch.exp(-time_diff_tensor * self.inv_temperature)

        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return g

    def forward_isolated(self, node_repr, prev_graph_embeds, time_diff_tensor):
        if self.self_loop:
            loop_message = torch.mm(node_repr, self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        # node_repr = node_repr + self.time_projection(prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature))
        node_repr = node_repr + torch.mm(prev_graph_embeds, self.time_weight) * torch.exp(-time_diff_tensor * self.inv_temperature)

        if self.bias:
            node_repr += self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        return node_repr
'''


class GRRGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=True, activation=None,
                 self_loop=True, dropout=0.0):
        super(GRRGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases,
                                          total_times, bias, activation, self_loop, dropout)
        self.post_aggregation = args.post_aggregation
        self.post_ensemble = args.post_ensemble
        self.num_layers = args.num_layers
        if args.type1:
            self.rnn = GRUCell(input_size=in_feat, hidden_size=out_feat)
        else:
            self.rnn = nn.GRU(input_size=in_feat, hidden_size=out_feat, num_layers=self.num_layers)

    def forward(self, g, prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes):
        result_graph, time_embedding = super().forward(g, time_batched_list_t, node_sizes)
        if self.learnable_lambda:
            # import pdb; pdb.set_trace()
            adjusted_prev_graph_embeds = self.decay_hidden(prev_graph_embeds, time_diff_tensor)
        else:
            adjusted_prev_graph_embeds = prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature)
        _, hidden = self.rnn(result_graph.ndata['h'].unsqueeze(0), adjusted_prev_graph_embeds.expand(self.num_layers, *prev_graph_embeds.shape))
        g.ndata['h'] = hidden[-1]
        if self.post_aggregation or self.post_ensemble or self.impute:
            return result_graph, g, time_embedding
        else:
            return g, time_embedding

    def forward_isolated(self, node_repr, prev_graph_embeds, time_diff_tensor, time):
        node_repr, time_embedding = super().forward_isolated(node_repr, time)
        # import pdb; pdb.set_trace()
        if self.learnable_lambda:
            adjusted_prev_graph_embeds = self.decay_hidden(prev_graph_embeds, time_diff_tensor)
        else:
            adjusted_prev_graph_embeds = prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature)
        _, hidden = self.rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds.expand(self.num_layers, *prev_graph_embeds.shape))

        if self.post_aggregation or self.post_ensemble or self.impute:
            return node_repr, hidden[-1], time_embedding
        else:
            return hidden[-1], time_embedding

    def forward_isolated_impute(self, node_repr, imputation_weight, prev_graph_embeds_loc, prev_graph_embeds_rec, time_diff_tensor, time):
        node_repr, time_embedding = super().forward_isolated(node_repr, time)
        node_repr = imputation_weight * prev_graph_embeds_loc + (1 - imputation_weight) * node_repr
        # import pdb; pdb.set_trace()
        if self.learnable_lambda:
            adjusted_prev_graph_embeds = self.decay_hidden(prev_graph_embeds_rec, time_diff_tensor)
        else:
            adjusted_prev_graph_embeds = prev_graph_embeds_rec * torch.exp(-time_diff_tensor * self.inv_temperature)

        _, hidden = self.rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds.expand(self.num_layers, *prev_graph_embeds_rec.shape))

        return hidden[-1], time_embedding



class RRGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=True, activation=None,
                 self_loop=True, dropout=0.0):
        super(RRGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases,
                                          total_times, bias, activation, self_loop, dropout)
        self.num_layers = args.num_layers
        self.time_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.time_weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, g, prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes):
        g = g.local_var()
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        node_repr = node_repr + torch.mm(prev_graph_embeds, self.time_weight) * torch.exp(-time_diff_tensor * self.inv_temperature)
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        time_embedding = self.get_time_embedding(time_batched_list_t, node_sizes)
        return g, time_embedding

    def forward_isolated(self, ent_embeds, prev_graph_embeds, time_diff_tensor, time):

        if self.self_loop:
            loop_message = torch.mm(ent_embeds, self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
            ent_embeds = ent_embeds + loop_message

        ent_embeds = ent_embeds + torch.mm(prev_graph_embeds, self.time_weight) * torch.exp(-time_diff_tensor * self.inv_temperature)
        if self.bias:
            ent_embeds += self.h_bias
        if self.activation:
            ent_embeds = self.activation(ent_embeds)
        time_embedding = self.time_embed[time]
        return ent_embeds, time_embedding


class RRGCN(nn.Module):
    # TODO: generalize to n layers
    def __init__(self, args, hidden_size, embed_size, num_rels, total_times):
        super(RRGCN, self).__init__()
        # in_feat = embed_size if static else hidden_size + embed_size
        self.rec_only_last_layer = args.rec_only_last_layer
        self.use_time_embedding = args.use_time_embedding
        # module = GRRGCNLayer if "GRRGCN" in args.module else RRGCNLayer
        module = {'GRRGCN': GRRGCNLayer, 'RRGCN': RRGCNLayer}[args.module]
        if not self.rec_only_last_layer:
            self.layer_1 = module(args, embed_size, hidden_size, 2 * num_rels, args.n_bases, total_times,
                                       bias=False, activation=None, self_loop=True, dropout=args.dropout)
        else:
            self.layer_1 = RGCNLayer(args, embed_size, hidden_size, 2 * num_rels, args.n_bases, total_times,
                                           bias=False, activation=None, self_loop=True, dropout=args.dropout)

        self.layer_2 = module(args, hidden_size, hidden_size, 2 * num_rels, args.n_bases, total_times,
                                   bias=False, activation=None, self_loop=True, dropout=args.dropout)
        self.impute = args.impute
        if self.impute:
            self.impute_weight = nn.Linear(1, 1)

    def forward(self, batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes):
        if not self.rec_only_last_layer:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, first_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes)
            if self.use_time_embedding:
                first_batched_graph.ndata['h'] = first_batched_graph.ndata['h'] + first_temp_embed
        else:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_batched_graph, second_temp_embed = self.layer_2(first_batched_graph, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes)

        if self.use_time_embedding:
            second_batched_graph.ndata['h'] = second_batched_graph.ndata['h'] + second_temp_embed
        return first_batched_graph.ndata['h'], second_batched_graph.ndata['h']

    def forward_isolated(self, ent_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time):
        if not self.rec_only_last_layer:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds, time_diff_tensor, time)
            if self.use_time_embedding:
                first_ent_embeds = first_ent_embeds + first_time_embedding
        else:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)

        second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated(first_ent_embeds, second_prev_graph_embeds, time_diff_tensor, time)
        if self.use_time_embedding:
            second_ent_embeds = second_ent_embeds + second_time_embedding
        return second_ent_embeds

    def forward_post_ensemble(self, batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes):
        if not self.rec_only_last_layer:
            _, first_batched_graph, first_temp_embed = self.layer_1(batched_graph, first_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes)
            if self.use_time_embedding:
                first_batched_graph.ndata['h'] = first_batched_graph.ndata['h'] + first_temp_embed
        else:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_local_graph, second_batched_graph, second_temp_embed = self.layer_2(first_batched_graph, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes)

        if self.use_time_embedding:
            second_local_graph.ndata['h'] = second_local_graph.ndata['h'] + second_temp_embed
            second_batched_graph.ndata['h'] = second_batched_graph.ndata['h'] + second_temp_embed

        return second_local_graph.ndata['h'], first_batched_graph.ndata['h'], second_batched_graph.ndata['h']

    def forward_post_ensemble_isolated(self, ent_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time, pre_embeds_loc):
        if not self.rec_only_last_layer:
            _, first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds, time_diff_tensor, time)
            if self.use_time_embedding:
                first_ent_embeds = first_ent_embeds + first_time_embedding
        else:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)

        second_local_embeds, second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated(first_ent_embeds, second_prev_graph_embeds, time_diff_tensor, time)

        if self.impute:
            imputation_weight = self.calc_impute_weight(time_diff_tensor)
            second_local_embeds = imputation_weight * pre_embeds_loc + (1 - imputation_weight) * second_local_embeds

        if self.use_time_embedding:
            second_local_embeds = second_local_embeds + second_time_embedding
            second_ent_embeds = second_ent_embeds + second_time_embedding

        return second_local_embeds, second_ent_embeds

    def forward_isolated_impute(self, ent_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time, pre_embeds_loc):
        if not self.rec_only_last_layer:
            _, first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds, time_diff_tensor, time)
            if self.use_time_embedding:
                first_ent_embeds = first_ent_embeds + first_time_embedding
        else:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)

        imputation_weight = self.calc_impute_weight(time_diff_tensor)
        second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated_impute(first_ent_embeds, imputation_weight, pre_embeds_loc, second_prev_graph_embeds, time_diff_tensor, time)

        if self.use_time_embedding:
            second_ent_embeds = second_ent_embeds + second_time_embedding

        return second_ent_embeds

    def calc_impute_weight(self, time_diff_tensor):
        return torch.exp(-torch.clamp(self.impute_weight(time_diff_tensor), min=0))