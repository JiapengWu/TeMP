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
        rel_id = graph.edges[x]['type_s'].item()
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
    edges_hist = defaultdict(list)
    for t, graph in enumerate(graphs):
        for u,v in graph.edges:
            rel_id = graph.edges[u,v]['type_s'].item()
            u_id = graph.nodes[u]['id'].item()
            v_id = graph.nodes[v]['id'].item()
            if rel_id >= num_rels:
                continue
            if stringtify:
                u_string = id2ent[u_id]
                v_string = id2ent[v_id]
                rel_id = id2rel[rel_id]
            edges_hist[(u_id, v_id)].append((t, rel_id))
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

def plot_node_edge_change_over_time():

    last_nodes = None
    last_edges = None

    entity_births = []
    entity_deaths = []
    entity_commons = []

    edge_births = []
    edge_deaths = []
    edge_commons = []

    new_subject_to_edge_count = defaultdict(int)
    new_object_to_edge_count = defaultdict(int)
    total_set = set()
    num_new_nodes = []
    k_ks = []
    u_ks = []
    u_us = []
    for i in times:
        train_graph = nx_train_graphs[i]
        val_graph = nx_val_graphs[i]
        test_graph = nx_test_graphs[i]
        node_idx = set([train_graph.nodes[n]['id'].item() for n in train_graph.nodes])
        new_nodes = node_idx.difference(total_set)
        total_set = total_set | new_nodes
        num_new_nodes.append(len(new_nodes))
        cur_edges = []
        nodes_from_edges = set()
        for u, v in train_graph.edges:
            rel_id = train_graph.edges[u, v]['type_s'].item()
            u_id = train_graph.nodes[u]['id'].item()
            v_id = train_graph.nodes[v]['id'].item()
            nodes_from_edges.add(u_id)
            nodes_from_edges.add(v_id)
            cur_edges.append((u_id, rel_id, v_id))
            if u_id in new_nodes:
                new_subject_to_edge_count[u_id] += 1
            if v_id in new_nodes:
                new_object_to_edge_count[v_id] += 1

        k_k = u_k = u_u = 0
        for u, v in val_graph.edges:
            # rel_id = val_graph.edges[u, v]['type_s'].item()
            u_id = val_graph.nodes[u]['id'].item()
            v_id = val_graph.nodes[v]['id'].item()
            u_unknown = u_id in new_nodes
            v_unknown = v_id in new_nodes
            if u_unknown and v_unknown:
                u_u += 1
            elif u_unknown or v_unknown:
                u_k += 1
            else:
                k_k += 1
        u_us.append(u_u)
        u_ks.append(u_k)
        k_ks.append(k_k)

        cur_edges = set(cur_edges)
        nodes_without_train_edges = new_nodes.difference(nodes_from_edges)
        # print(len(nodes_without_train_edges))
        # if len(nodes_without_train_edges) > 0:
        #     pdb.set_trace()
        for n in nodes_without_train_edges:
            new_subject_to_edge_count[n] = new_object_to_edge_count[n] = 0

        if last_edges:
            common = cur_edges.intersection(last_edges)
            edge_deaths.append(len(last_edges) - len(common))
            edge_births.append(len(cur_edges) - len(common))
            edge_commons.append(len(common))
        else:
            edge_commons.append(0)
            edge_births.append(len(cur_edges))

        if last_nodes:
            common = node_idx.intersection(last_nodes)
            entity_deaths.append(len(last_nodes) - len(common))
            entity_births.append(len(node_idx) - len(common))
            entity_commons.append(len(common))
        else:
            entity_commons.append(0)
            entity_births.append(len(node_idx))
        last_edges = cur_edges
        last_nodes = node_idx


    # subject_freq_to_count = defaultdict(int)
    # object_freq_to_count = defaultdict(int)

    # for eid, freq in new_subject_to_edge_count.items():
    #     subject_freq_to_count[freq] += 1
    # for eid, freq in new_object_to_edge_count.items():
    #     object_freq_to_count[freq] += 1
    # subject_freq_to_count = {k: v for k, v in sorted(subject_freq_to_count.items(), key=lambda item: item[0])}
    # object_freq_to_count = {k: v for k, v in sorted(object_freq_to_count.items(), key=lambda item: item[0])}

    plt.plot(entity_commons, label='# entities in common with t - 1')
    plt.plot(entity_births, label='# entities added to t - 1')
    plt.plot(entity_deaths, label='# entities deleted at t + 1')
    plt.plot(num_new_nodes, label='# new entities')
    plt.legend()
    plt.xlabel("time t")
    plt.ylabel("# entities")
    plt.savefig(os.path.join(fig_path, "entity_statistics.png"))
    plt.clf()

    plt.plot(edge_commons, label='# edges in common with t - 1')
    plt.plot(edge_births, label='# edges added to t - 1')
    plt.plot(edge_deaths, label='# edges deleted at t + 1')
    plt.legend()
    plt.xlabel("time t")
    plt.ylabel("# edges")
    plt.savefig(os.path.join(fig_path, "edge_statistics.png"))
    plt.clf()

    plt.plot(k_ks, label='# both entities are known')
    plt.plot(u_ks, label='# one entity is known')
    plt.plot(u_us, label='# both entities are unknown')
    sum_kk = np.sum(k_ks)
    sum_uk = np.sum(u_ks)
    sum_uu = np.sum(u_us)
    sum_all = sum_kk + sum_uk + sum_uu
    print(sum_kk / sum_all)
    print(sum_uk / sum_all)
    print(sum_uu / sum_all)

    plt.legend()
    plt.xlabel("time t")
    plt.ylabel("# validation edges")
    plt.savefig(os.path.join(fig_path, "val_edge_statistics.png"))
    plt.clf()
    # plt.scatter(list(subject_freq_to_count.keys()), list(subject_freq_to_count.values()), label='new entity is subject')
    # plt.scatter(list(object_freq_to_count.keys()), list(object_freq_to_count.values()), label='new entity is object')

    # pdb.set_trace()

    subject_freq = list(new_subject_to_edge_count.values())
    object_freq = list(new_object_to_edge_count.values())
    plt.hist(subject_freq, bins=max(subject_freq), label='new entity is subject')
    plt.hist(object_freq, bins=max(object_freq), label='new entity is object')
    # print(subject_freq_to_count)
    # print(object_freq_to_count)
    plt.legend()
    plt.xlabel("number of new edges with new entity")
    plt.ylabel("number of entities")
    plt.savefig(os.path.join(fig_path, "new_entity_statistics.png"))
    plt.clf()


def plot_tail_change_over_time(graphs, stringtify=False):

    # construct a dict edges -> relation type
    head_rel_hist = defaultdict(list)
    for t, graph in enumerate(graphs):
        for u, v in graph.edges:
            rel_id = graph.edges[u,v]['type_s'].item()
            u_id = graph.nodes[u]['id'].item()
            v_id = graph.nodes[v]['id'].item()
            if rel_id >= num_rels:
                continue
            if stringtify:
                u_string = id2ent[u_id]
                v_string = id2ent[v_id]
                rel_string = id2rel[rel_id]
            head_rel_hist[(u_id, rel_id)].append((t, v_id))

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

def draw_graphs():

    # print(nx_graphs)
    graph = nx_train_graphs[0].subgraph(list(nx_train_graphs[0].nodes)[:100])
    node_lables, edge_labels = stringtify_graph_nodes_edges(graph, id2ent, id2rel)
    print("Nodes: {}".format(node_lables))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=10)

    # nx.draw_networkx_labels(graph, pos, dict(zip(pos, node_lables)))
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()

# def get_all_nodes():
#     total_set = set()
#     for i in times:
#         train_graph = nx_train_graphs[i]
#         val_graph = nx_val_graphs[i]
#         test_graph = nx_test_graphs[i]
#
#         train_node_idx = set([train_graph.nodes[n]['id'].item() for n in train_graph.nodes])
#         val_node_idx = set([train_graph.nodes[n]['id'].item() for n in val_graph.nodes])
#         test_node_idx = set([train_graph.nodes[n]['id'].item() for n in test_graph.nodes])
#         node_idx = train_node_idx | val_node_idx | test_node_idx
#
#         total_set = total_set | node_idx
#     pdb.set_trace()


if __name__ == '__main__':
    args = process_args()
    train_data, train_times = load_quadruples(args.dataset, 'train.txt')
    max_time_step = len(train_times)
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    id2ent, id2rel = id2entrel(args.dataset, num_rels)
    # id2ent = id2rel = None
    train_graph_dict, val_graph_dict, test_graph_dict = build_interpolation_graphs(args)
    times = list(train_graph_dict.keys())
    nx_train_graphs = [train_graph_dict[i].to_networkx(edge_attrs=['type_s'], node_attrs=['id']) for i in times]
    nx_val_graphs = [val_graph_dict[i].to_networkx(edge_attrs=['type_s'], node_attrs=['id']) for i in times]
    nx_test_graphs = [test_graph_dict[i].to_networkx(edge_attrs=['type_s'], node_attrs=['id']) for i in times]
    # draw_graphs()

    # exit()
    fig_path = os.path.join('figs', args.dataset_dir + "_" + args.dataset.split('/')[-1])
    # import pdb; pdb.set_trace()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # plot_num_facts_nodes_over_time(nx_graphs)
    # plot_tail_change_over_time(nx_graphs, False)
    plot_node_edge_change_over_time()

    # edges_hist = calc_hist(nx_graphs, id2ent, id2rel, False)
    # plot_rel_over_time(edges_hist)
    # pdb.set_trace()
    #
    # for graph in nx_graphs:
    #     calc_num_edges_statistics_for_t(graph)
    #     calc_num_facts_per_rel(graph)


