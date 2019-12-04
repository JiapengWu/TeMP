from utils.dataset import *
from utils.args import process_args
import pdb
import networkx as nx
import matplotlib.pyplot as plt
# from models.TKG_VRE import VKG_VAE


def stringtify_graph_nodes_edges(graph, id2ent, id2rel):
    node_lables = [id2ent[n] for n in graph.nodes]
    edge_labels = {}
    for x in graph.edges:
        rel_id = min(graph.edges[x]['type_s'].item(), graph.edges[x]['type_o'].item())
        # pdb.set_trace()
        edge_labels[x] = id2rel[rel_id]

    return node_lables, edge_labels


def calc_num_edges_statistics_for_t(graph):
    print("Number of nodes: {}".format(len(graph.nodes)))
    degrees = list(dict(graph.in_degree()).values())
    plt.hist(degrees, bins=np.max(degrees) - np.min(degrees))
    plt.xlabel("degree per entity")
    plt.ylabel("# entities")
    plt.savefig(os.path.join(fig_path, "degree_per_entity.png"))
    plt.clf()


def calc_num_facts_per_rel(graph):
    rel_count = {}
    for x in graph.edges:
        rel_id = min(graph.edges[x]['type_s'].item(), graph.edges[x]['type_o'].item())
        try:
            rel_count[rel_id] += 1
        except:
            rel_count[rel_id] = 1
    print("Number of relations: {}".format(len(rel_count.keys())))
    print("Number of edges: {}".format(len(graph.edges)))
    rel_counts = list(rel_count.values())
    plt.hist(rel_counts, bins=np.max(rel_counts) - np.min(rel_counts))

    plt.savefig(os.path.join(fig_path, "num_facts_per_rel.png"))
    plt.clf()


def calc_hist(graphs, id2ent, id2rel, stringtify=False):
    # construct a dict edges -> relation type
    edges_hist = {}
    for t, graph in enumerate(graphs):
        for u,v in graph.edges:

            rel_id = graph.edges[u,v]['type_s'].item()
            if rel_id >= num_rels:
                continue
            if stringtify:
                u = id2ent[u]
                v = id2ent[v]
                rel_id = id2rel[rel_id]
            try:
                edges_hist[(u, v)].append((t, rel_id))
            except:
                edges_hist[(u, v)] = [(t, rel_id)]
    return edges_hist


def plot_rel_over_time(edges_hist):
    # edge_stats = {}
    mutate_intervals = []
    repeat_intervals = []
    num_rel_types = []
    hist_events = []
    for edge, hist in edges_hist.items():
        # print(edge)
        # print(hist)
        # head = id2ent[edge[0]]
        # tail = id2ent[edge[1]]
        # rel_set = [id2rel[rel] for t, rel in hist]

        prev_rel = -1
        prev_t = -1
        mutate_invertal = []
        repeat_invertal = []
        rels = []
        for t, rel in hist:
            rels.append(rel)
            if prev_t == -1:
                prev_t = t; prev_rel = rel
                continue
            if rel != prev_rel:
                mutate_invertal.append(t - prev_t)
            else:
                repeat_invertal.append(t - prev_t)
            prev_t = t; prev_rel = rel
        rels = list(set(rels))
        hist_events.append(len(hist))
        num_rel_types.append(len(rels))
        mutate_intervals.extend(mutate_invertal)
        repeat_intervals.extend(repeat_invertal)
        # edge_stats[edge] = [mutate_invertal, repeat_invertal, rels]
        # pdb.set_trace()
    plt.hist(mutate_intervals, bins=np.max(mutate_intervals) - np.min(mutate_intervals))
    plt.xlabel("mutate intervals")
    plt.ylabel("# intervals")
    plt.savefig(os.path.join(fig_path, "rel_mutation_interval.png"))
    plt.clf()
    plt.hist(repeat_intervals, bins=np.max(repeat_intervals) - np.min(repeat_intervals))
    plt.xlabel("repeat intervals")
    plt.ylabel("# intervals")
    plt.savefig(os.path.join(fig_path, "rel_repeat_interval.png"))
    plt.clf()
    plt.hist(hist_events, bins=np.max(hist_events) - np.min(hist_events))
    plt.xlabel("hist events")
    plt.ylabel("# entity pairs")
    plt.savefig(os.path.join(fig_path, "num_hist_event_per_ent_pair.png"))
    plt.clf()
    plt.hist(num_rel_types, bins=np.max(num_rel_types) - np.min(num_rel_types))
    plt.xlabel("relation types")
    plt.ylabel("# entity pairs")
    plt.savefig(os.path.join(fig_path, "num_hist_rel_types_per_ent_pair.png"))
    plt.clf()

def plot_node_change_over_time(graphs):
    node_seq = [set(list(graph.nodes)) for graph in graphs]
    last_nodes = None
    births = []
    deaths = []
    commons = []
    for nodes in node_seq:
        if last_nodes:
            deaths.append(len(last_nodes - nodes))
            births.append(len(nodes - last_nodes))
            commons.append(len(nodes.intersection(last_nodes)))
        last_nodes = nodes
    plt.plot(births)
    plt.xlabel("time")
    plt.ylabel("# entity births")
    plt.savefig(os.path.join(fig_path, "num_entity_births.png"))
    plt.clf()
    plt.plot(deaths)
    plt.xlabel("time")
    plt.ylabel("# entity deaths")
    plt.savefig(os.path.join(fig_path, "num_entity_deaths.png"))
    plt.clf()
    plt.plot(commons)
    plt.xlabel("time")
    plt.ylabel("# entity commons")
    plt.savefig(os.path.join(fig_path, "num_entity_commons.png"))
    plt.clf()


def plot_edge_over_time(graphs):
    edge_seq = [set(list(graph.edges)) for graph in graphs]
    last_edges = None
    births = []
    deaths = []
    commons = []
    for edges in edge_seq:
        if last_edges:
            deaths.append(len(last_edges - edges))
            births.append(len(edges - last_edges))
            commons.append(len(edges.intersection(last_edges)))
        last_edges = edges
    plt.plot(births)
    plt.xlabel("time")
    plt.ylabel("# edge births")
    plt.savefig(os.path.join(fig_path, "num_edge_births.png"))
    plt.clf()
    plt.plot(deaths)
    plt.xlabel("time")
    plt.ylabel("# edge deaths")
    plt.savefig(os.path.join(fig_path, "num_edge_deaths.png"))
    plt.clf()
    plt.plot(commons)
    plt.xlabel("time")
    plt.ylabel("# edge commons")
    plt.savefig(os.path.join(fig_path, "num_edge_commons.png"))
    plt.clf()


def plot_tail_change_over_time(graphs, stringtify=False):

    # construct a dict edges -> relation type
    head_rel_hist = {}
    for t, graph in enumerate(graphs):
        for u,v in graph.edges:
            rel_id = graph.edges[u,v]['type_s'].item()
            if rel_id >= num_rels:
                continue
            if stringtify:
                u = id2ent[u]
                v = id2ent[v]
                rel_id = id2rel[rel_id]
            try:
                head_rel_hist[(u, rel_id)].append((t, v))
            except:
                head_rel_hist[(u, rel_id)] = [(t, v)]

    mutate_intervals = []
    concurrents = []
    repeat_intervals = []
    num_rel_types = []
    hist_events = []

    for head_rel, hist in head_rel_hist.items():

        # print(head_rel)
        # print(hist)
        # pdb.set_trace()
        prev_tail = -1
        prev_t = -1
        mutate_invertal = []
        repeat_invertal = []
        tails = []
        concurrent = 0
        for t, tail in hist:
            tails.append(tail)
            if prev_t == -1:
                prev_t = t; prev_tail = tail
                continue
            if prev_t != t:
                if tail != prev_tail:
                    mutate_invertal.append(t - prev_t)
                else:
                    repeat_invertal.append(t - prev_t)
            else:
                concurrent += 1
            prev_t = t; prev_tail = tail
        concurrents.append(concurrent)
        tails = list(set(tails))
        hist_events.append(len(hist))
        num_rel_types.append(len(tails))
        mutate_intervals.extend(mutate_invertal)
        repeat_intervals.extend(repeat_invertal)

    plt.hist(mutate_intervals, bins=np.max(mutate_intervals) - np.min(mutate_intervals))
    plt.xlabel("mutate intervals")
    plt.ylabel("# intervals")
    plt.savefig(os.path.join(fig_path, "tail_mutate_interval.png"))
    plt.clf()
    plt.hist(repeat_intervals, bins=np.max(repeat_intervals) - np.min(repeat_intervals) + 1)
    plt.hist(repeat_intervals)
    plt.xlabel("repeat intervals")
    plt.ylabel("# intervals")
    plt.savefig(os.path.join(fig_path, "tail_repeate_interval.png"))
    plt.clf()
    plt.hist(concurrents, bins=np.max(concurrents) - np.min(concurrents))
    plt.xlabel("# concurrent events")
    plt.ylabel("# head-rel pairs")
    plt.savefig(os.path.join(fig_path, "num_concurrent_events_per_head_rel.png"))
    plt.clf()
    plt.hist(hist_events, bins=np.max(hist_events) - np.min(hist_events))
    plt.xlabel("# hist events")
    plt.ylabel("# head-rel pairs")
    plt.savefig(os.path.join(fig_path, "num_hist_events_per_head_rel.png"))
    plt.clf()
    plt.hist(num_rel_types, bins=np.max(num_rel_types) - np.min(num_rel_types))
    plt.xlabel("# tail types")
    plt.ylabel("# head-rel pairs")
    plt.savefig(os.path.join(fig_path, "num_tail_types_per_head_rel.png"))
    plt.clf()


def plot_num_facts_nodes_over_time(graphs):
    num_nodes = []
    num_facts = []
    for graph in graphs:
        num_nodes.append((len(list(graph.nodes))))
        num_facts.append((len(list(graph.edges))))

    plt.plot(num_nodes)
    plt.xlabel("time")
    plt.ylabel("# nodes")
    plt.savefig(os.path.join(fig_path, "num_nodes_over_time.png"))
    plt.clf()
    plt.plot(num_facts)
    plt.xlabel("time")
    plt.ylabel("# facts")
    plt.savefig(os.path.join(fig_path, "num_edges_over_time.png"))
    plt.clf()


if __name__ == '__main__':
    args = process_args()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    if args.dataset == 'extrapolation/ICEWS14':
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'test.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'test.txt')
    else:
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'valid.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'valid.txt','test.txt')
    # id2ent, id2rel = id2entrel(args.dataset, num_rels)
    id2ent = id2rel = None
    train_graph_dict = build_extrapolation_time_stamp_graph(args)
    times = list(train_graph_dict.keys())
    nx_graphs = [train_graph_dict[i].to_networkx(edge_attrs=['type_s', 'type_o']) for i in times]
    fig_path = os.path.join('figs', args.dataset.split('/')[-1])
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    # print(nx_graphs)
    # graph = nx_graphs[0].subgraph(list(nx_graphs[0].nodes)[:100])
    # node_lables, edge_labels = stringtify_graph_nodes_edges(graph, id2ent, id2rel)
    # print("Nodes: {}".format(node_lables))
    plot_num_facts_nodes_over_time(nx_graphs)
    plot_tail_change_over_time(nx_graphs, False)
    plot_edge_over_time(nx_graphs)
    plot_node_change_over_time(nx_graphs)

    edges_hist = calc_hist(nx_graphs, id2ent, id2rel, False)
    plot_rel_over_time(edges_hist)
    # pdb.set_trace()
    #
    # for graph in nx_graphs:
    #     calc_num_edges_statistics_for_t(graph)
    #     calc_num_facts_per_rel(graph)


    # pos = nx.spring_layout(graph)
    # nx.draw_networkx_nodes(graph, pos, node_size=10)
    # nx.draw_networkx_labels(graph, pos, dict(zip(pos, node_lables)))
    # nx.draw_networkx_edges(graph, pos, alpha=0.5)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
