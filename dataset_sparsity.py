from utils.dataset import *
from utils.args import process_args
import pdb
import networkx as nx
import matplotlib.pyplot as plt
# from models.TKG_VRE import VKG_VAE


def plot_sparsity():
    fig_path = os.path.join('figs', args.dataset_dir + "_" + args.dataset.split('/')[-1])
    # import pdb; pdb.set_trace()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    node_nums = []
    sum_hists = []
    exist_hists = []
    for tim in total_times:
        node_num, sum_hist, exist_hist = get_num_occurrence(tim)
        node_nums.append(node_num)
        sum_hists.append(sum_hist)
        exist_hists.append(exist_hist)

    # plt.figure(figsize=(3, 9/4), dpi=400, facecolor='w', edgecolor='k')
    x = list(range(len(node_nums)))
    plt.bar(x, node_nums, label='# active entities at each time step')
    plt.bar(x, exist_hists, label='# active entities occurred recently')
    plt.xlabel("time step")
    plt.ylabel("# entities")
    plt.legend()
    plt.savefig(os.path.join(fig_path, "num_existing_hist_nodes.png"), bbox_inches="tight")
    plt.clf()

    plt.plot(sum_hists, label='avg # of temporal facts with shared pattern')
    plt.xlabel("time step")
    plt.ylabel("# entities")
    # plt.legend()
    plt.savefig(os.path.join(fig_path, "num_avg_active_hist_nodes.png"), bbox_inches="tight")

def plot_edge_sparsity():
    fig_path = os.path.join('figs', args.dataset_dir + "_" + args.dataset.split('/')[-1])
    # import pdb; pdb.set_trace()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    edge_nums = []
    sum_hists = []
    exist_hists = []
    for tim in total_times:
        edge_num, sum_hist, exist_hist = get_hist_edge_num(tim)
        edge_nums.append(edge_num)
        sum_hists.append(sum_hist)
        exist_hists.append(exist_hist)

    plt.plot(edge_nums, label='# active nodes')
    plt.plot(exist_hists, label='# existing active nodes')
    plt.xlabel("time")
    plt.ylabel("# nodes")
    plt.legend()
    plt.savefig(os.path.join(fig_path, "num_existing_hist_nodes.png"))
    plt.clf()

    plt.plot(sum_hists, label='# avg active nodes')
    plt.xlabel("time")
    plt.ylabel("# nodes")
    plt.legend()
    plt.savefig(os.path.join(fig_path, "num_avg_active_hist_nodes.png"))

def print_example_kg():
    entities_1 = ["United States"]
    entities_2 = ["France"]

    for t in times:
        pre_graph = nx_train_graphs[t]
        for u, v in pre_graph.edges:
            rel_id = pre_graph.edges[u, v]['type_s'].item()
            if rel_id >= num_rels:
                continue
            u_id = pre_graph.nodes[u]['id'].item()
            v_id = pre_graph.nodes[v]['id'].item()

            u_string = id2ent[u_id]
            v_string = id2ent[v_id]
            u_match = v_match = False
            print_string = False

            for e in entities_1:
                if e in u_string:
                    u_match = True
                if e in v_string:
                    v_match = True
            if u_match:
                for e in entities_2:
                    if e in v_string:
                        print_string = True
            if v_match:
                for e in entities_2:
                    if e in u_string:
                        print_string = True

            if print_string:
                print(u_string, id2rel[rel_id], v_string, t)

def get_hist_edge_num(tim):
    idx2count = dict()

    print(tim)
    cur_idx = train_graph_dict[tim].ids.values()
    for id in cur_idx:
        idx2count[id] = 0
    cur_graph = nx_train_graphs[tim]
    for u, v in cur_graph.edges:
        u_id = cur_graph.nodes[u]['id'].item()
        v_id = cur_graph.nodes[u]['id'].item()
        rel_id = cur_graph.edges[u, v]['type_s'].item()
    for t in range(max(0, int(tim/interval) - args.train_seq_len), int(tim/interval)):
        try:
            pre_graph = nx_train_graphs[t]
        except:
            continue
        for u, v in pre_graph.edges:
            # rel_id = pre_graph.edges[u, v]['type_s'].item()
            # if rel_id >= num_rels:
            #     continue
            u_id = pre_graph.nodes[u]['id'].item()
            v_id = pre_graph.nodes[u]['id'].item()
            rel_id = pre_graph.edges[u, v]['type_s'].item()

            if u_id in cur_idx:
                idx2count[u_id] += 1

    idx2exist = np.sum(np.array(list(idx2count.values())) > 0)
    return len(idx2count), np.average(list(idx2count.values())), idx2exist


def get_num_occurrence(tim):
    idx2count = dict()

    # print(tim)
    cur_idx = train_graph_dict[tim].ids.values()
    for id in cur_idx:
        idx2count[id] = 0

    for t in range(max(0, int(tim/interval) - args.train_seq_len), int(tim/interval)):
        try:
            pre_graph = nx_train_graphs[t]
        except:
            continue
        for u, v in pre_graph.edges:
            # rel_id = pre_graph.edges[u, v]['type_s'].item()
            # if rel_id >= num_rels:
            #     continue
            u_id = pre_graph.nodes[u]['id'].item()

            if u_id in cur_idx:
                idx2count[u_id] += 1

    idx2exist = np.sum(np.array(list(idx2count.values())) > 0)
    return len(idx2count), np.average(list(idx2count.values())), idx2exist


def calc_entity_hist():
    for tim in total_times:
        print(tim)
        # graph = nx_graphs[tim]
        # pdb.set_trace()
        cur_idx = train_graph_dict[tim].ids.values()
        entity_hist = defaultdict(lambda: defaultdict(list))

        for t in range(max(0, tim - args.train_seq_len, tim)):
            try:
                pre_graph = nx_train_graphs[t]
            except:
                continue
            for u, v in pre_graph.edges:
                rel_id = pre_graph.edges[u, v]['type_s'].item()
                if rel_id >= num_rels:
                    continue
                u_id = pre_graph.nodes[u]['id'].item()
                v_id = pre_graph.nodes[v]['id'].item()

                u_string = id2ent[u_id]
                if u_id in cur_idx:
                    v_string = id2ent[v_id]
                    rel_string = id2rel[rel_id]
                    entity_hist[u_string][t].append((rel_string, v_string))
                else:
                    if t == tim: pdb.set_trace()
        if tim >= 15:
            pretty_print_hist(entity_hist, tim)
            pdb.set_trace()


def calc_entity_hist_future():
    for tim in total_times:
        print(tim)
        # graph = nx_graphs[tim]
        # pdb.set_trace()
        cur_idx = train_graph_dict[tim].ids.values()
        entity_hist = defaultdict(lambda: defaultdict(list))

        for t in range(max(0, tim - int(args.train_seq_len / 2), min(len(total_times), tim + int(args.train_seq_len / 2)))):
            if t == tim: continue
            try:
                pre_graph = nx_train_graphs[t]
            except:
                continue
            for u, v in pre_graph.edges:
                rel_id = pre_graph.edges[u, v]['type_s'].item()
                if rel_id >= num_rels:
                    continue
                u_id = pre_graph.nodes[u]['id'].item()
                v_id = pre_graph.nodes[v]['id'].item()

                u_string = id2ent[u_id]
                if u_id in cur_idx:
                    v_string = id2ent[v_id]
                    rel_string = id2rel[rel_id]
                    entity_hist[u_string][t].append((rel_string, v_string))
        if tim >= 15:
            pretty_print_hist(entity_hist, tim)
            pdb.set_trace()


def pretty_print_hist(entity_hist, tim):
    train_triples, val_triples, test_triples = group_current_triples(tim)

    for entity in entity_hist.keys():
        if len(test_triples[entity]) == 0 and len(val_triples[entity]) == 0:
            continue
        print(entity)
        for time in entity_hist[entity]:
            print("At time {}: ".format(time))
            for tup in entity_hist[entity][time]:
                print("{}\t{}\t{}".format(entity, *tup))
        print("Train triples")
        for tup in train_triples[entity]:
            print("{}\t{}\t{}".format(entity, *tup))
        print()
        if len(val_triples[entity]) != 0:
            print("Validation triples: ")
            for tup in val_triples[entity]:
                print("{}\t{}\t{}".format(entity, *tup))

        if len(test_triples[entity]) != 0:
            print("Test triples: ")
            for tup in test_triples[entity]:
                print("{}\t{}\t{}".format(entity, *tup))

        print()
    print()


def group_current_triples(tim):
    train_graph = nx_train_graphs[tim]
    val_graph = nx_val_graphs[tim]
    test_graph = nx_test_graphs[tim]
    train_triples = defaultdict(list)
    val_triples = defaultdict(list)
    test_triples = defaultdict(list)
    for graph, triples in zip([train_graph, val_graph, test_graph], [train_triples, val_triples, test_triples]):
        for u, v in graph.edges:
            rel_id = graph.edges[u, v]['type_s'].item()
            u_id = graph.nodes[u]['id'].item()
            v_id = graph.nodes[v]['id'].item()
            u_string = id2ent[u_id]
            v_string = id2ent[v_id]
            rel_string = id2rel[rel_id]
            triples[u_string].append((rel_string, v_string))
    return train_triples, val_triples, test_triples


def group_current_triples_ent_rel(tim):
    train_graph = nx_train_graphs[tim]
    val_graph = nx_val_graphs[tim]
    test_graph = nx_test_graphs[tim]
    train_triples_s_r = defaultdict(list)
    val_triples_s_r = defaultdict(list)
    test_triples_s_r = defaultdict(list)
    train_triples_o_r = defaultdict(list)
    val_triples_o_r = defaultdict(list)
    test_triples_o_r = defaultdict(list)
    for graph, triples_s_r, triples_o_r in zip([train_graph, val_graph, test_graph], [train_triples_s_r, val_triples_s_r, test_triples_s_r], [train_triples_o_r, val_triples_o_r, test_triples_o_r]):
        for u, v in graph.edges:
            rel_id = graph.edges[u, v]['type_s'].item()
            u_id = graph.nodes[u]['id'].item()
            v_id = graph.nodes[v]['id'].item()
            u_string = id2ent[u_id]
            v_string = id2ent[v_id]
            rel_string = id2rel[rel_id]
            triples_s_r[(u_string, rel_string)].append((v_string))
            triples_o_r[(v_string, rel_string)].append((u_string))
    return train_triples_s_r, val_triples_s_r, test_triples_s_r, train_triples_o_r, val_triples_o_r, test_triples_o_r


def show_hist_facts():
    for tim in total_times:
        # print()
        print(tim)
        val_graph = nx_val_graphs[tim]
        test_graph = nx_test_graphs[tim]
        for graph in val_graph, test_graph:
            for u, v in graph.edges:
                rel_id = graph.edges[u, v]['type_s'].item()
                u_id = graph.nodes[u]['id'].item()
                v_id = graph.nodes[v]['id'].item()
                u_string = id2ent[u_id]
                v_string = id2ent[v_id]
                rel_string = id2rel[rel_id]

                if tim >= 15:
                    print()
                    print("Validation triple: {}\t{}\t{}".format(u_string, rel_string, v_string))
                for t in range(max(0, tim - args.train_seq_len, tim+1)):
                    try:
                        pre_graph = nx_train_graphs[t]
                    except:
                        continue
                    for u, v in pre_graph.edges:
                        cur_rel_id = pre_graph.edges[u, v]['type_s'].item()
                        cur_u_id = pre_graph.nodes[u]['id'].item()
                        cur_v_id = pre_graph.nodes[v]['id'].item()

                        cur_rel_string = id2rel[cur_rel_id]
                        cur_u_string = id2ent[cur_u_id]
                        cur_v_string = id2ent[cur_v_id]

                        if tim >= 15:
                            if (cur_u_id == u_id or cur_v_id == u_id) and cur_rel_id == rel_id:
                                print("At time {}: {}\t{}\t{}".format(t, cur_u_string, cur_rel_string, cur_v_string))
                            if (cur_v_id == v_id or cur_u_id == v_id) and cur_rel_id == rel_id:
                                print("At time {}: {}\t{}\t{}".format(t, cur_u_string, cur_rel_string, cur_v_string))

        if tim >= 15:
            pdb.set_trace()


def plot_node_change_over_time(graphs):
    last_nodes = None
    births = []
    deaths = []
    commons = []
    for graph in graphs:
        node_idx = set([graph.nodes[n]['id'].item() for n in graph.nodes])
        if last_nodes:
            common = node_idx.intersection(last_nodes)
            deaths.append(len(last_nodes) - len(common))
            births.append(len(node_idx) - len(common))
            commons.append(len(common))
            # deaths.append(len(last_nodes - node_idx))
            # births.append(len(node_idx - last_nodes))
            # commons.append(len(node_idx.intersection(last_nodes)))
        last_nodes = node_idx

    plt.plot(births)
    plt.xlabel("time t")
    plt.ylabel("# entities added to t - 1")
    plt.savefig(os.path.join(fig_path, "num_entity_births.png"))
    plt.clf()
    plt.plot(deaths)
    plt.xlabel("time t")
    plt.ylabel("# entities deleted from t - 1")
    plt.savefig(os.path.join(fig_path, "num_entity_deaths.png"))
    plt.clf()
    plt.plot(commons)
    plt.xlabel("time t")
    plt.ylabel("# entities in common with t - 1")
    plt.savefig(os.path.join(fig_path, "num_entity_commons.png"))
    plt.clf()


if __name__ == '__main__':
    args = process_args()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    # id2ent = id2rel = None
    id2ent, id2rel = id2entrel(args.dataset, num_rels)
    fig_path = os.path.join('figs', args.dataset_dir + "_" + args.dataset.split('/')[-1])
    if args.dataset_dir == 'interpolation':
        train_graph_dict, val_graph_dict, test_graph_dict = build_interpolation_graphs(args)
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'valid.txt', 'test.txt')
    else:
        train_graph_dict, val_graph_dict, test_graph_dict = build_extrapolation_time_stamp_graph(args)
        total_times = list(train_graph_dict.keys())
    times = list(train_graph_dict.keys())
    nx_train_graphs = [train_graph_dict[i].to_networkx(edge_attrs=['type_s'], node_attrs=['id']) for i in times]
    nx_val_graphs = [val_graph_dict[i].to_networkx(edge_attrs=['type_s'], node_attrs=['id']) for i in times]
    nx_test_graphs = [test_graph_dict[i].to_networkx(edge_attrs=['type_s'], node_attrs=['id']) for i in times]
    # pdb.set_trace()

    interval = total_times[1] - total_times[0]
    print_example_kg()
    # plot_sparsity()
    # calc_entity_hist()
    # calc_entity_hist_future()
    # show_hist_facts()