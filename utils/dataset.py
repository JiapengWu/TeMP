import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch


def load_quadruples(dataset_path, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(dataset_path, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(dataset_path, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(dataset_path, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_total_number(dataset_path, fileName="stat.txt"):
    with open(os.path.join(dataset_path, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    # if uniq_v.max() + 1 != len(uniq_v):
    #     import pdb; pdb.set_trace()

    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


def build_time_stamp_graph(args):

    graph_dict_path = os.path.join(args.dataset, 'train_graphs.txt')

    if not os.path.isfile(graph_dict_path):
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        # test_data, test_times = load_quadruples('', 'test.txt')
        # dev_data, dev_times = load_quadruples('', 'valid.txt')
        num_e, num_r = get_total_number(args.dataset, 'stat.txt')

        graph_dict_train = {}

        for tim in train_times:
            print(str(tim) + '\t' + str(max(train_times)))
            data = get_data_with_t(train_data, tim)
            graph_dict_train[tim] = get_big_graph(data, num_r)

            with open(graph_dict_path, 'wb') as fp:
                pickle.dump(graph_dict_train, fp)
    else:
        with open(graph_dict_path, 'rb') as f:
            graph_dict_train = pickle.load(f)

    return graph_dict_train


def id2entrel(dataset_path, num_rels):
    id2ent = {}; id2rel = {}
    with open(os.path.join(dataset_path, "entity2id.txt"), 'r') as fr:
        for line in fr:
            line_split = line.strip().split("\t")
            name = line_split[0]
            id = int(line_split[1])
            id2ent[id] = name
    with open(os.path.join(dataset_path, "relation2id.txt"), 'r') as fr:
        for line in fr:
            line_split = line.strip().split("\t")
            name = line_split[0]
            id = int(line_split[1])
            id2rel[id] = name
            id2rel[id + num_rels] = "{}_inv".format(name)
    return id2ent, id2rel