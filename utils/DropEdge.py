from utils.dataset import load_quadruples
from utils.frequency import count_freq_per_time, calc_aggregated_statistics, temp_func
import os
from collections import defaultdict
import numpy as np
import torch
from utils.utils import comp_deg_norm, node_norm_to_edge_norm
import pdb
from utils.args import process_args
import time
from utils.dataset import build_interpolation_graphs

class DropEdge():
    def __init__(self, args, graph_dict_train, graph_dict_val, graph_dict_test):
        self.args = args
        self.train_seq_len = self.args.train_seq_len
        self.train_data, self.train_times = load_quadruples(args.dataset, 'train.txt')
        self.future = "Bi" in args.module
        self.graph_dict_train = graph_dict_train
        self.graph_dict_val = graph_dict_val
        self.graph_dict_test = graph_dict_test
        self.max_time_step = len(self.train_times)
        self.count_frequency()
        self.lower = args.rate_lower
        self.upper = args.rate_upper
        assert self.upper > self.lower
        self.diff = self.upper - self.lower
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.lambda_3 = self.args.lambda_3
        # self.drop_rate_cache = defaultdict(lambda: defaultdict(list))
        # self.pre_cal_drop_rate()

    def count_frequency(self):
        self.triple_freq_per_time_step, self.ent_pair_freq_per_time_step, self.sub_freq_per_time_step, self.obj_freq_per_time_step, \
            self.rel_freq_per_time_step, self.sub_rel_freq_per_time_step, self.obj_rel_freq_per_time_step = count_freq_per_time(self.train_data)

        # self.triple_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        # self.ent_pair_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        # self.sub_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        # self.obj_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        # self.rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        # self.sub_rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        # self.obj_rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
        #
        # for quad in self.train_data:
        #     sub, rel, obj, tim = tuple(quad)
        #     self.triple_freq_per_time_step[tim][(sub, rel, obj)] += 1
        #     self.ent_pair_freq_per_time_step[tim][(sub, obj)] += 1
        #     self.sub_freq_per_time_step[tim][sub] += 1
        #     self.obj_freq_per_time_step[tim][obj] += 1
        #     self.rel_freq_per_time_step[tim][rel] += 1
        #     self.sub_rel_freq_per_time_step[tim][(sub, rel)] += 1
        #     self.obj_rel_freq_per_time_step[tim][(obj, rel)] += 1

        self.triple_freq_per_time_step_agg = defaultdict(temp_func)
        self.ent_pair_freq_per_time_step_agg = defaultdict(temp_func)
        self.sub_freq_per_time_step_agg = defaultdict(temp_func)
        self.obj_freq_per_time_step_agg = defaultdict(temp_func)
        self.rel_freq_per_time_step_agg = defaultdict(temp_func)
        self.sub_rel_freq_per_time_step_agg = defaultdict(temp_func)
        self.obj_rel_freq_per_time_step_agg = defaultdict(temp_func)

        for target_time in self.train_times:
            triples = list(self.triple_freq_per_time_step[target_time].keys())
            ent_pairs = list(self.ent_pair_freq_per_time_step[target_time].keys())
            subs = list(self.sub_freq_per_time_step[target_time].keys())
            objs = list(self.obj_freq_per_time_step[target_time].keys())
            rels = list(self.rel_freq_per_time_step[target_time].keys())
            sub_rels = list(self.sub_rel_freq_per_time_step[target_time].keys())
            obj_rels = list(self.obj_rel_freq_per_time_step[target_time].keys())
            upper = target_time if not self.future else min(self.max_time_step + 1, target_time + self.train_seq_len)
            for cur_time in range(max(0, target_time - self.train_seq_len + 1), upper):

                if cur_time == target_time: continue
                calc_aggregated_statistics(self.triple_freq_per_time_step_agg, triples, self.triple_freq_per_time_step, target_time, cur_time)
                calc_aggregated_statistics(self.ent_pair_freq_per_time_step_agg, ent_pairs, self.ent_pair_freq_per_time_step, target_time, cur_time)
                calc_aggregated_statistics(self.sub_freq_per_time_step_agg, subs, self.sub_freq_per_time_step, target_time, cur_time)
                calc_aggregated_statistics(self.obj_freq_per_time_step_agg, objs, self.obj_freq_per_time_step, target_time, cur_time)
                calc_aggregated_statistics(self.rel_freq_per_time_step_agg, rels, self.rel_freq_per_time_step, target_time, cur_time)
                calc_aggregated_statistics(self.sub_rel_freq_per_time_step_agg, sub_rels, self.sub_rel_freq_per_time_step, target_time, cur_time)
                calc_aggregated_statistics(self.obj_rel_freq_per_time_step_agg, obj_rels, self.obj_rel_freq_per_time_step, target_time, cur_time)

    def calc_dropout_prob(self, t_src, rel, t_dst, cur_time, target_time):
        target_triple_freq = self.triple_freq_per_time_step_agg[target_time]
        target_ent_pair_freq = self.ent_pair_freq_per_time_step_agg[target_time]
        target_sub_rel_freq = self.sub_rel_freq_per_time_step_agg[target_time]
        target_obj_rel_freq = self.obj_rel_freq_per_time_step_agg[target_time]

        target_triple_lst = list(target_triple_freq.keys())
        target_ent_pair_lst = list(target_ent_pair_freq.keys())
        target_sub_rel_lst = list(target_sub_rel_freq.keys())
        target_obj_rel_lst = list(target_obj_rel_freq.keys())
        drop_rate_lst = self.drop_rate_cache[target_time][cur_time]
        for s, r, o in zip(t_src, rel, t_dst):
            s = s.item(); r = r.item(); o = o.item()
            if (s, r, o) in target_triple_lst:
                rate = self.lower + self.diff * (1 - self.lambda_1 / (target_triple_freq[(s, r, o)] + self.lambda_1))
            elif (s, o) in target_ent_pair_lst:
                rate = self.lower + self.diff * (1 - self.lambda_2 / (target_ent_pair_freq[(s, o)] + self.lambda_2))
            elif (s, r) in target_sub_rel_lst:
                rate = self.lower + self.diff * (1 - self.lambda_3 / (target_sub_rel_freq[(s, r)] + self.lambda_3))
            elif (o, r) in target_obj_rel_lst:
                rate = self.lower + self.diff * (1 - self.lambda_3 / (target_obj_rel_freq[(o, r)] + self.lambda_3))
            else:
                rate = self.lower
            drop_rate_lst.append(rate)

    def pre_cal_drop_rate(self):
        self.drop_rate_cache = defaultdict(lambda: defaultdict(list))
        for target_time in self.train_times:
            upper = target_time if not self.future else min(self.max_time_step, target_time + self.train_seq_len)
            for cur_time in range(max(0, target_time - self.train_seq_len + 1), upper):
                # pdb.set_trace()
                if cur_time == target_time: continue
                cur_g = self.graph_dict_train[cur_time]
                src, rel, dst = cur_g.edges()[0], cur_g.edata['type_s'], cur_g.edges()[1]
                t_src = torch.tensor([cur_g.ids[s.item()] for s in src])
                t_dst = torch.tensor([cur_g.ids[d.item()] for d in dst])
                self.calc_dropout_prob(t_src, rel, t_dst, cur_time, target_time)

    def sample_subgraph(self, cur_time, target_time):
        # sampled_graph_list = []
        # upper = target_time if not self.future else min(self.max_time_step, target_time + self.train_seq_len)
        # for cur_time in range(max(0, target_time - self.train_seq_len + 1), upper):
        cur_g = self.graph_dict_train[cur_time]
        src, rel, dst = cur_g.edges()[0], cur_g.edata['type_s'], cur_g.edges()[1]
        drop_rates = self.drop_rate_cache[target_time][cur_time]
        # pdb.set_trace()
        mask = torch.bernoulli(1 - torch.tensor(drop_rates)) == 1
        sampled_idx = torch.arange(src.shape[0])[mask]
        sg = cur_g.edge_subgraph(sampled_idx, preserve_nodes=True)
        node_norm = comp_deg_norm(sg)
        sg.ndata.update({'id': cur_g.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
        sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
        sg.edata['type_s'] = rel[sampled_idx]
        sg.ids = cur_g.ids
        return sg


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)

    graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)
    drop_edge = DropEdge(args, graph_dict_train, graph_dict_val, graph_dict_test)
