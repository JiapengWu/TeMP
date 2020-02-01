import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.RGCN import RGCNLayer, RGCNBlockLayer


class RRGCNLayer(RGCNLayer):
    def __init__(self, args, in_feat, out_feat, bias=None, activation=None,
                 self_loop=True, dropout=0.0):
        super(RRGCNLayer, self).__init__(in_feat, out_feat, bias=None, activation=None,
                 self_loop=True, dropout=0.0)
        self.num_layers = args.num_layers
        # self.rnn = LSTMCell(input_size=in_feat, hidden_size=out_feat)
        self.inv_temperature = args.inv_temperature
        self.rnn = nn.GRU(input_size=in_feat, hidden_size=out_feat, num_layers=self.num_layers, dropout=args.dropout)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_graph_embeds, time_diff_tensor=None):
        # for g in g_list:
        g = g.local_var()
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)
        # apply bias and activation
        node_repr = g.ndata['h']

        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        # import pdb; pdb.set_trace()
        adjusted_prev_graph_embeds = prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature)
        _, hidden = self.rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds.expand(self.num_layers, *prev_graph_embeds.shape))

        if self.activation:
            hidden = self.activation(hidden)
        g.ndata['h'] = hidden[-1]
        return g

    def forward_isolated(self, ent_embeds, prev_graph_embeds, time_diff_tensor=None):
        # import pdb; pdb.set_trace()
        # ent_embeds = ent_embeds + torch.mm(prev_graph_embeds, self.time_weight) * torch.exp(-time_diff_tensor / self.temperature)
        if self.self_loop:
            loop_message = torch.mm(ent_embeds, self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
            ent_embeds = ent_embeds + loop_message
        if self.bias:
            ent_embeds += self.bias
        if self.activation:
            ent_embeds = self.activation(ent_embeds)
        adjusted_prev_graph_embeds = prev_graph_embeds * torch.exp(-time_diff_tensor * self.inv_temperature)
        _, hidden = self.rnn(ent_embeds.unsqueeze(0), adjusted_prev_graph_embeds.expand(self.num_layers, *prev_graph_embeds.shape))
        return ent_embeds


class RRGCNBlockLayer(RRGCNLayer):
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RRGCNBlockLayer, self).__init__(args, in_feat, out_feat, bias,
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


class RRGCN(nn.Module):
    # TODO: generalize to n layers
    def __init__(self, args, hidden_size, embed_size, num_rels, static=False):
        super(RRGCN, self).__init__()
        # in_feat = embed_size if static else hidden_size + embed_size
        self.rec_only_last_layer = args.rec_only_last_layer

        if not self.rec_only_last_layer:
            self.layer_1 = RRGCNBlockLayer(args, embed_size, hidden_size, 2 * num_rels, args.n_bases,
                                           activation=None, self_loop=True, dropout=args.dropout)
        else:
            self.layer_1 = RGCNBlockLayer(embed_size, hidden_size, 2 * num_rels, args.n_bases,
                                           activation=None, self_loop=True, dropout=args.dropout)

        self.layer_2 = RRGCNBlockLayer(args, hidden_size, hidden_size, 2 * num_rels, args.n_bases,
                                       activation=F.relu, self_loop=True, dropout=args.dropout)

    def forward(self, batched_graph, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor=None):
        if not self.rec_only_last_layer:
            batched_graph = self.layer_1(batched_graph, first_prev_graph_embeds, time_diff_tensor)
        else:
            batched_graph = self.layer_1(batched_graph)
        return batched_graph, self.layer_2(batched_graph, second_prev_graph_embeds, time_diff_tensor)

    def forward_isolated(self, ent_embeds, first_prev_graph_embeds, second_prev_graph_embeds, time_diff_tensor=None):
        if not self.rec_only_last_layer:
            ent_embeds = self.layer_1.forward_isolated(ent_embeds, first_prev_graph_embeds, time_diff_tensor)
        else:
            ent_embeds = self.layer_1.forward_isolated(ent_embeds)

        return self.layer_2.forward_isolated(ent_embeds, second_prev_graph_embeds, time_diff_tensor)
