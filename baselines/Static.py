import dgl
from baselines.TKG_Non_Recurrent import TKG_Non_Recurrent


class Static(TKG_Non_Recurrent):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(Static, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        self.negative_rate = self.args.negative_rate

    def get_per_graph_ent_embeds(self, t_list, g_list):
        batched_graph = dgl.batch(g_list)
        ent_embeds = self.ent_embeds[batched_graph.ndata['id']].view(-1, self.embed_size)
        node_sizes = [len(g.nodes()) for g in g_list]
        return ent_embeds.split(node_sizes)
