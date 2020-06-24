import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.RGCN import RGCNLayer
from models.GRU_cell import GRUCell
import pdb

class BiGRRGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=None, activation=None,
                 self_loop=True, dropout=0.0):
        super(BiGRRGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases,
                                            total_times, bias, activation, self_loop, dropout)
        self.num_layers = args.num_layers

        # module = {"BiRRGCN": nn.RNN, "BiGRRGCN": nn.GRU}[args.module]
        self.post_aggregation = args.post_aggregation
        self.post_ensemble = args.post_ensemble

        if args.type1:
            self.forward_rnn = GRUCell(input_size=in_feat, hidden_size=out_feat)
            self.backward_rnn = GRUCell(input_size=in_feat, hidden_size=out_feat)
        else:
            self.forward_rnn = nn.GRU(input_size=in_feat, hidden_size=out_feat, num_layers=self.num_layers)
            self.backward_rnn = nn.GRU(input_size=in_feat, hidden_size=out_feat, num_layers=self.num_layers)

    def forward(self, g, prev_graph_embeds_forward, time_diff_tensor_forward, prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes):
        result_graph, time_embedding = super().forward(g, time_batched_list_t, node_sizes)

        if self.learnable_lambda:
            adjusted_prev_graph_embeds_forward = self.decay_hidden(prev_graph_embeds_forward, time_diff_tensor_forward)
        else:
            adjusted_prev_graph_embeds_forward = prev_graph_embeds_forward * torch.exp(-time_diff_tensor_forward * self.inv_temperature)
        _, hidden_forward = self.forward_rnn(result_graph.ndata['h'].unsqueeze(0), adjusted_prev_graph_embeds_forward.expand(self.num_layers, *prev_graph_embeds_forward.shape))

        if self.learnable_lambda:
            adjusted_prev_graph_embeds_backward = self.decay_hidden(prev_graph_embeds_backward, time_diff_tensor_backward)
        else:
            adjusted_prev_graph_embeds_backward = prev_graph_embeds_backward * torch.exp(-time_diff_tensor_backward * self.inv_temperature)
        _, hidden_backward = self.backward_rnn(result_graph.ndata['h'].unsqueeze(0), adjusted_prev_graph_embeds_backward.expand(self.num_layers, *prev_graph_embeds_backward.shape))
        # adding x itself? Like this:
        # g.ndata['h'] = hidden_forward[-1] + hidden_backward[-1] + result_graph.ndata['h']
        g.ndata['h'] = hidden_forward[-1] + hidden_backward[-1]
        if self.post_aggregation or self.post_ensemble or self.impute:
            return result_graph, g, time_embedding
        else:
            return g, time_embedding

    def forward_one_direction(self, g, prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward):
        result_graph, time_embedding = super().forward(g, time_batched_list_t, node_sizes)
        model = self.forward_rnn if forward else self.backward_rnn

        if self.learnable_lambda:
            adjusted_prev_graph_embeds = self.decay_hidden(prev_graph_embeds, time_diff_tensor)
        else:
            adjusted_prev_graph_embeds = prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature)
        _, hidden = model(result_graph.ndata['h'].unsqueeze(0), adjusted_prev_graph_embeds.expand(self.num_layers, *prev_graph_embeds.shape))

        g.ndata['h'] = hidden[-1]
        if self.post_aggregation or self.post_ensemble or self.impute:
            return result_graph, g, time_embedding
        else:
            return g, time_embedding

    def forward_isolated(self, node_repr, prev_graph_embeds_forward, prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time):
        node_repr, time_embedding = super().forward_isolated(node_repr, time)
        if self.learnable_lambda:
            adjusted_prev_graph_embeds_forward = self.decay_hidden(prev_graph_embeds_forward, time_diff_tensor_forward)
            adjusted_prev_graph_embeds_backward = self.decay_hidden(prev_graph_embeds_backward, time_diff_tensor_backward)
        else:
            adjusted_prev_graph_embeds_forward = prev_graph_embeds_forward * torch.exp(-time_diff_tensor_forward * self.inv_temperature)
            adjusted_prev_graph_embeds_backward = prev_graph_embeds_backward * torch.exp(-time_diff_tensor_backward * self.inv_temperature)

        _, hidden_forward = self.forward_rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds_forward.expand(self.num_layers, *prev_graph_embeds_forward.shape))

        _, hidden_backward = self.backward_rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds_backward.expand(self.num_layers, *prev_graph_embeds_backward.shape))

        if self.post_aggregation or self.post_ensemble or self.impute:
            return node_repr, hidden_forward[-1] + hidden_backward[-1], time_embedding
        else:
            return hidden_forward[-1] + hidden_backward[-1], time_embedding

    def forward_isolated_impute(self, node_repr, imputation_weight_forward, imputation_weight_backward, second_embeds_forward_loc, second_embeds_backward_loc,
                                prev_graph_embeds_forward, prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time):
        node_repr, time_embedding = super().forward_isolated(node_repr, time)

        node_repr = imputation_weight_forward * second_embeds_forward_loc + imputation_weight_backward * second_embeds_backward_loc + (
                    1 - imputation_weight_forward - imputation_weight_backward) * node_repr
        if self.learnable_lambda:
            adjusted_prev_graph_embeds_forward = self.decay_hidden(prev_graph_embeds_forward, time_diff_tensor_forward)
            adjusted_prev_graph_embeds_backward = self.decay_hidden(prev_graph_embeds_backward, time_diff_tensor_backward)
        else:
            adjusted_prev_graph_embeds_forward = prev_graph_embeds_forward * torch.exp(-time_diff_tensor_forward * self.inv_temperature)
            adjusted_prev_graph_embeds_backward = prev_graph_embeds_backward * torch.exp(-time_diff_tensor_backward * self.inv_temperature)


        _, hidden_forward = self.forward_rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds_forward.expand(self.num_layers, *prev_graph_embeds_forward.shape))
        _, hidden_backward = self.backward_rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds_backward.expand(self.num_layers, *prev_graph_embeds_backward.shape))

        return hidden_forward[-1] + hidden_backward[-1], time_embedding

class BiRRGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=None, activation=None,
                 self_loop=True, dropout=0.0):
        super(BiRRGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases,
                                            total_times, bias, activation, self_loop, dropout)
        self.num_layers = args.num_layers

        self.time_weight_forward = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.time_weight_forward, gain=nn.init.calculate_gain('relu'))

        self.time_weight_backward = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.time_weight_backward, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, prev_graph_embeds_forward, time_diff_tensor_forward, prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes):
        g = g.local_var()
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)
        # apply bias and activation
        adjusted_prev_graph_embeds_forward = prev_graph_embeds_forward * torch.exp(-time_diff_tensor_forward * self.inv_temperature)
        adjusted_prev_graph_embeds_backward = prev_graph_embeds_backward * torch.exp(-time_diff_tensor_backward * self.inv_temperature)

        node_repr = g.ndata['h']
        node_repr = node_repr + torch.mm(adjusted_prev_graph_embeds_forward, self.time_weight_forward)
        node_repr = node_repr + torch.mm(adjusted_prev_graph_embeds_backward, self.time_weight_backward)

        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        time_embedding = self.get_time_embedding(time_batched_list_t, node_sizes)
        return g, time_embedding

    def forward_one_direction(self, g, prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward):
        g = g.local_var()
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        node_repr = g.ndata['h']
        weight = self.time_weight_forward if forward else self.time_weight_backward
        node_repr = node_repr + torch.mm(prev_graph_embeds, weight) * torch.exp(-time_diff_tensor * self.inv_temperature)

        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        time_embedding = self.get_time_embedding(time_batched_list_t, node_sizes)
        return g, time_embedding

    def forward_isolated(self, node_repr, prev_graph_embeds_forward, prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time):

        if self.self_loop:
            loop_message = torch.mm(node_repr, self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
            node_repr = node_repr + loop_message

        adjusted_prev_graph_embeds_forward = prev_graph_embeds_forward * torch.exp(-time_diff_tensor_forward * self.inv_temperature)
        adjusted_prev_graph_embeds_backward = prev_graph_embeds_backward * torch.exp(-time_diff_tensor_backward * self.inv_temperature)

        node_repr = node_repr + torch.mm(adjusted_prev_graph_embeds_forward, self.time_weight_forward)
        node_repr = node_repr + torch.mm(adjusted_prev_graph_embeds_backward, self.time_weight_backward)
        if self.bias:
            node_repr += self.h_bias
        if self.activation:
            node_repr = self.activation(node_repr)
        time_embedding = self.time_embed[time]

        return node_repr, time_embedding


class BiRRGCN(nn.Module):
    # TODO: generalize to n layers
    def __init__(self, args, hidden_size, embed_size, num_rels, total_times):
        super(BiRRGCN, self).__init__()
        self.rec_only_last_layer = args.rec_only_last_layer
        self.use_time_embedding = args.use_time_embedding
        module = {'BiGRRGCN': BiGRRGCNLayer, 'BiRRGCN': BiRRGCNLayer}[args.module]
        if not self.rec_only_last_layer:
            self.layer_1 = module(args, embed_size, hidden_size, 2 * num_rels, args.n_bases, total_times,
                                         bias=False, activation=None, self_loop=True, dropout=args.dropout)
        else:

            self.layer_1 = RGCNLayer(args, embed_size, hidden_size, 2 * num_rels, args.n_bases, total_times,
                                       bias=False, activation=None, self_loop=True, dropout=args.dropout)
        self.layer_2 = module(args, hidden_size, hidden_size, 2 * num_rels, args.n_bases, total_times,
                                     bias=False, activation=F.relu, self_loop=True, dropout=args.dropout)

        self.impute = args.impute
        if self.impute:
            self.impute_weight_forward = nn.Linear(1, 1)
            self.impute_weight_backward = nn.Linear(1, 1)

    def forward(self, batched_graph, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward, first_prev_graph_embeds_backward,
                second_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes):

        if not self.rec_only_last_layer:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, first_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                                 first_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes)
            if self.use_time_embedding:
                first_batched_graph.ndata['h'] = first_batched_graph.ndata['h'] + first_temp_embed
        else:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_batched_graph, second_temp_embed = self.layer_2(first_batched_graph, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                               second_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes)

        if self.use_time_embedding:
            second_batched_graph.ndata['h'] = second_batched_graph.ndata['h'] + second_temp_embed
        return second_batched_graph.ndata['h']

    def forward_one_direction(self, batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward):
        if not self.rec_only_last_layer:
            first_batched_graph, first_temp_embed = self.layer_1.forward_one_direction(batched_graph, first_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward)
            if self.use_time_embedding:
                first_batched_graph.ndata['h'] = first_batched_graph.ndata['h'] + first_temp_embed
        else:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_batched_graph, second_temp_embed = self.layer_2.forward_one_direction(first_batched_graph, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward)

        if self.use_time_embedding:
            second_batched_graph.ndata['h'] = second_batched_graph.ndata['h'] + second_temp_embed
        return first_batched_graph.ndata['h'], second_batched_graph.ndata['h']

    def forward_isolated(self, ent_embeds, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                           first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, time):

        if not self.rec_only_last_layer:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds_forward,
                                                                                   first_prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time)
            if self.use_time_embedding:
                first_ent_embeds = first_ent_embeds + first_time_embedding
        else:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)

        second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated(first_ent_embeds, second_prev_graph_embeds_forward,
                                                                                 second_prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time)
        if self.use_time_embedding:
            second_ent_embeds = second_ent_embeds + second_time_embedding
        return second_ent_embeds

    def forward_post_ensemble(self, batched_graph, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward, first_prev_graph_embeds_backward,
                second_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes):

        if not self.rec_only_last_layer:
            _, first_batched_graph, first_temp_embed = self.layer_1(batched_graph, first_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                                 first_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes)
            if self.use_time_embedding:
                first_batched_graph.ndata['h'] = first_batched_graph.ndata['h'] + first_temp_embed
        else:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_local_graph, second_batched_graph, second_temp_embed = self.layer_2(first_batched_graph, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                                               second_prev_graph_embeds_backward, time_diff_tensor_backward, time_batched_list_t, node_sizes)

        if self.use_time_embedding:
            second_local_graph.ndata['h'] = second_local_graph.ndata['h'] + second_temp_embed
            second_batched_graph.ndata['h'] = second_batched_graph.ndata['h'] + second_temp_embed

        return second_local_graph.ndata['h'], second_batched_graph.ndata['h']

    def forward_post_ensemble_one_direction(self, batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward):
        if not self.rec_only_last_layer:
            _, first_batched_graph, first_temp_embed = self.layer_1.forward_one_direction(batched_graph, first_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward)
            if self.use_time_embedding:
                first_batched_graph.ndata['h'] = first_batched_graph.ndata['h'] + first_temp_embed
        else:
            first_batched_graph, first_temp_embed = self.layer_1(batched_graph, time_batched_list_t, node_sizes)

        second_local_graph, second_batched_graph, second_temp_embed = self.layer_2.forward_one_direction(first_batched_graph, second_prev_graph_embeds, time_diff_tensor, time_batched_list_t, node_sizes, forward)

        if self.use_time_embedding:
            second_local_graph.ndata['h'] = second_local_graph.ndata['h'] + second_temp_embed
            second_batched_graph.ndata['h'] = second_batched_graph.ndata['h'] + second_temp_embed

        return second_local_graph.ndata['h'], first_batched_graph.ndata['h'], second_batched_graph.ndata['h']

    def forward_post_ensemble_isolated(self, ent_embeds, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                           first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, time, second_embeds_forward_loc, second_embeds_backward_loc):

        if not self.rec_only_last_layer:
            _, first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds_forward,
                                                                                   first_prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time)
            if self.use_time_embedding:
                first_ent_embeds = first_ent_embeds + first_time_embedding
        else:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)

        second_local_embeds, second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated(first_ent_embeds, second_prev_graph_embeds_forward,
                                                                                 second_prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time)

        if self.impute:
            # pdb.set_trace()
            imputation_weight_forward = torch.exp(-torch.clamp(self.impute_weight_forward(time_diff_tensor_forward), min=0))/2
            imputation_weight_backward = torch.exp(-torch.clamp(self.impute_weight_backward(time_diff_tensor_backward), min=0))/2
            second_local_embeds = imputation_weight_forward * second_embeds_forward_loc + imputation_weight_backward * second_embeds_backward_loc + (1 - imputation_weight_forward - imputation_weight_backward) * second_local_embeds
            # pdb.set_trace()
        if self.use_time_embedding:
            second_local_embeds = second_local_embeds + second_time_embedding
            second_ent_embeds = second_ent_embeds + second_time_embedding
        return second_local_embeds, second_ent_embeds

    def forward_isolated_impute(self, ent_embeds, first_prev_graph_embeds_forward, second_prev_graph_embeds_forward, time_diff_tensor_forward,
                                first_prev_graph_embeds_backward, second_prev_graph_embeds_backward, time_diff_tensor_backward, time, second_embeds_forward_loc, second_embeds_backward_loc):
        if not self.rec_only_last_layer:
            _, first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds_forward,
                                                                                   first_prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time)
            if self.use_time_embedding:
                first_ent_embeds = first_ent_embeds + first_time_embedding
        else:
            first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)
        imputation_weight_forward = torch.exp(-torch.clamp(self.impute_weight_forward(time_diff_tensor_forward), min=0)) / 2
        imputation_weight_backward = torch.exp(-torch.clamp(self.impute_weight_backward(time_diff_tensor_backward), min=0)) / 2

        second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated_impute(first_ent_embeds, imputation_weight_forward, imputation_weight_backward, second_embeds_forward_loc, second_embeds_backward_loc,
                                                                                 second_prev_graph_embeds_forward, second_prev_graph_embeds_backward, time_diff_tensor_forward, time_diff_tensor_backward, time)

        if self.use_time_embedding:
            second_ent_embeds = second_ent_embeds + second_time_embedding

        return second_ent_embeds
