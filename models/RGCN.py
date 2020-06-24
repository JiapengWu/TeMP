import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb

class RGCNLayer(nn.Module):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=True,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        # self.use_time_embedding = args.use_time_embedding
        self.time_embed = nn.Parameter(torch.Tensor(len(total_times), in_feat))
        nn.init.xavier_uniform_(self.time_embed, gain=nn.init.calculate_gain('relu'))

        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.in_feat = in_feat
        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.bias == True:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout) if dropout else None
        self.inv_temperature = args.inv_temperature
        self.learnable_lambda = args.learnable_lambda
        if self.learnable_lambda:
            self.exponential_decay = nn.Linear(1, 1)
        self.impute = args.impute

    def get_time_embedding(self, time_batched_list_t, node_sizes):
        time_embedding = []
        for t, size in zip(time_batched_list_t, node_sizes):
            time_embedding.append(self.time_embed[t].unsqueeze(0).expand(size, self.in_feat))
        return torch.cat(time_embedding, dim=0)

    def forward(self, g, time_batched_list_t, node_sizes):
        # pdb.set_trace()
        g = g.local_var()
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        # print(time_batched_list_t)
        # print(node_sizes)
        time_embedding = self.get_time_embedding(time_batched_list_t, node_sizes)
        return g, time_embedding

    def forward_isolated(self, ent_embeds, time):
        if self.self_loop:
            loop_message = torch.mm(ent_embeds, self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
            ent_embeds = ent_embeds + loop_message
        if self.bias:
            ent_embeds += self.h_bias
        if self.activation:
            ent_embeds = self.activation(ent_embeds)
        time_embedding = self.time_embed[time]
        return ent_embeds, time_embedding

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type_s']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

    def decay_hidden(self, prev_graph_embeds, time_diff_tensor):
        return prev_graph_embeds * torch.exp(-torch.clamp(self.exponential_decay(time_diff_tensor), min=0))


'''
class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type_s']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
'''

class RGCN(nn.Module):
    def __init__(self, args, hidden_size, embed_size, num_rels, total_times):
        super(RGCN, self).__init__()
        self.use_time_embedding = args.use_time_embedding
        self.layer_1 = RGCNLayer(args, embed_size, hidden_size, 2 * num_rels, args.n_bases,
                   total_times, activation=None, self_loop=True, dropout=args.dropout)
        self.layer_2 = RGCNLayer(args, hidden_size, hidden_size, 2 * num_rels, args.n_bases,
                   total_times, activation=F.relu, self_loop=True, dropout=args.dropout)

    def forward(self, batched_graph, time_batched_list_t, node_sizes):
        first_batch_graph, first_time_embedding = self.layer_1(batched_graph, time_batched_list_t, node_sizes)
        second_batch_graph, second_time_embedding = self.layer_2(first_batch_graph, time_batched_list_t, node_sizes)
        if self.use_time_embedding:
            second_batch_graph.ndata['h'] = second_batch_graph.ndata['h'] + second_time_embedding
        return second_batch_graph

    def forward_isolated(self, ent_embeds, time):
        first_ent_embeds, first_time_embedding = self.layer_1.forward_isolated(ent_embeds, time)
        second_ent_embeds, second_time_embedding = self.layer_2.forward_isolated(first_ent_embeds, time)
        return second_ent_embeds + second_time_embedding if self.use_time_embedding else second_ent_embeds
