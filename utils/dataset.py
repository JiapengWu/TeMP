import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
from torch.utils.data import Dataset
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
    train_graph_dict_path = os.path.join(args.dataset, 'train_graphs.txt')
    dev_graph_dict_path = os.path.join(args.dataset, 'dev_graphs.txt')
    test_graph_dict_path = os.path.join(args.dataset, 'test_graphs.txt')

    graph_dict_train = {}
    graph_dict_dev = {}
    graph_dict_test = {}
    if not os.path.isfile(train_graph_dict_path) or not os.path.isfile(dev_graph_dict_path) or not os.path.isfile(test_graph_dict_path):
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        if args.dataset == 'ICEWS14':
            dev_data, dev_times = load_quadruples(args.dataset, 'test.txt')
        else:
            dev_data, dev_times = load_quadruples(args.dataset, 'valid.txt')
        num_e, num_r = get_total_number(args.dataset, 'stat.txt')


        for times, datas, path, graph_dict in zip([
            train_times, dev_times, test_times],
                [train_data, dev_data, test_data],
                [train_graph_dict_path, dev_graph_dict_path, test_graph_dict_path],
                [graph_dict_train, graph_dict_dev, graph_dict_test]
        ):
            for tim in times:
                print(str(tim) + '\t' + str(max(times)))
                data = get_data_with_t(datas, tim)
                graph_dict[tim] = get_big_graph(data, num_r)
                with open(path, 'wb') as fp:
                    pickle.dump(graph_dict, fp)

    else:
        graph_dicts = []
        for path in train_graph_dict_path, dev_graph_dict_path, test_graph_dict_path:
            with open(path, 'rb') as f:
                graph_dicts.append(pickle.load(f))
        graph_dict_train, graph_dict_dev, graph_dict_test = graph_dicts
    return graph_dict_train, graph_dict_dev, graph_dict_test


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


class TimeDataset(Dataset):
    def __init__(self, times):
        self.times = times

    def __getitem__(self, index):
        return self.times[index]

    def __len__(self):
        return len(self.times)