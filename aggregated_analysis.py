import sys
from link_prediction_analysis import construct_ref_data, count_entity_freq_per_train_graph
import pickle
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from utils.dataset import *
import pdb
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import argparse
# import seaborn as sns; sns.set(color_codes=True)
from sklearn.preprocessing import PolynomialFeatures


def get_args():
    parser = argparse.ArgumentParser(description='TKG-VAE')
    parser.add_argument("path_1", type=str)
    parser.add_argument("path_2", type=str)
    parser.add_argument("path_3", type=str)
    parser.add_argument("path_4", type=str)
    # parser.add_argument("path_5", type=str)
    parser.add_argument("--bidirectional", action='store_true')
    # parser.add_argument("--bidirectional", action='store_true')

    return parser.parse_args()


def key_based_sort_dict(inp):
    return {k: v for k, v in sorted(inp.items(), key=lambda item: item[0])}


def obj_metric_freq(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_obj_rel_counts, freq_obj_counts, freq_rel_counts):
    triple_count = 0
    obj_count = 0
    rel_count = 0
    obj_rel_count = 0
    ent_pair_count = 0
    upper = time + 1 if not bidirectional else min(max_time_step + 1, time + train_seq_len)
    for t in range(max(0, time - train_seq_len + 1), upper):
        if t == time: continue
        triple_count += triple_freq_per_time_step[(s, r, o)][t]
        ent_pair_count += ent_pair_freq_per_time_step[(s, o)][t]
        obj_rel_count += obj_rel_freq_per_time_step[(o, r)][t]
        obj_count += obj_freq_per_time_step[o][t]
        rel_count += rel_freq_per_time_step[r][t]

    freq_triple_counts[triple_count].append(rank)
    freq_ent_pair_counts[ent_pair_count].append(rank)
    freq_obj_rel_counts[obj_rel_count].append(rank)
    freq_obj_counts[obj_count].append(rank)
    freq_rel_counts[rel_count].append(rank)


def sub_metric_freq(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_sub_counts, freq_rel_counts):
    triple_count = 0
    sub_count = 0
    rel_count = 0
    sub_rel_count = 0
    ent_pair_count = 0
    upper = time + 1 if not bidirectional else min(max_time_step + 1, time + train_seq_len)
    for t in range(max(0, time - train_seq_len + 1), upper):
        if t == time: continue
        triple_count += triple_freq_per_time_step[(s, r, o)][t]
        ent_pair_count += ent_pair_freq_per_time_step[(s, o)][t]
        sub_rel_count += sub_rel_freq_per_time_step[(s, r)][t]
        sub_count += sub_freq_per_time_step[s][t]
        rel_count += rel_freq_per_time_step[r][t]

    freq_triple_counts[triple_count].append(rank)
    freq_ent_pair_counts[ent_pair_count].append(rank)
    freq_sub_rel_counts[sub_rel_count].append(rank)
    freq_sub_counts[sub_count].append(rank)
    freq_rel_counts[rel_count].append(rank)


def obj_repeat_freq(r, o, time, rank, freq_obj_rel_counts, freq_obj_counts):
    obj_count = 0
    obj_rel_count = 0
    upper = time + 1 if not bidirectional else min(max_time_step + 1, time + train_seq_len)
    for t in range(max(0, time - train_seq_len + 1), upper):
        if t == time: continue
        obj_rel_count += obj_rel_freq_per_time_step[(o, r)][t]
        obj_count += obj_freq_per_time_step[o][t]

    freq_obj_rel_counts[obj_rel_count].append(rank)
    freq_obj_counts[obj_count].append(rank)

def sub_repeat_freq(s, r, time, rank, freq_sub_rel_counts, freq_sub_counts):
    sub_count = 0
    sub_rel_count = 0
    upper = time + 1 if not bidirectional else min(max_time_step + 1, time + train_seq_len)
    for t in range(max(0, time - train_seq_len + 1), upper):
        if t == time: continue
        sub_rel_count += sub_rel_freq_per_time_step[(s, r)][t]
        sub_count += sub_freq_per_time_step[s][t]

    freq_sub_rel_counts[sub_rel_count].append(rank)
    freq_sub_counts[sub_count].append(rank)


def group_ranks(metric):
    # pdb.set_trace()
    metric = key_based_sort_dict(metric)
    # pdb.set_trace()
    if "gdelt" in dataset:
        threshold = 300
    else:
        threshold = 50
    result = {}
    accumulated_size = 0
    accumulated_freqs = []
    accumulated_ranks = []
    # print([len(ranks) for ranks in metric.values()])
    for freq, ranks in metric.items():

        accumulated_size += len(ranks)
        accumulated_freqs.append(freq)
        accumulated_ranks.extend(ranks)
        if accumulated_size > threshold:
            result[np.mean(accumulated_freqs)] = accumulated_ranks
            accumulated_size = 0
            accumulated_freqs = []
            accumulated_ranks = []
    return result


def calc_hit_10_per_score(predictions):
    freq_sub_rel_counts = defaultdict(list)
    freq_obj_rel_counts = defaultdict(list)
    freq_sub_counts = defaultdict(list)
    freq_obj_counts = defaultdict(list)

    freq_sub_rel_counts_metric = dict()
    freq_obj_rel_counts_metric = dict()
    freq_sub_counts_metric = dict()
    freq_obj_counts_metric = dict()

    freq_sub_rel_counts_per_rank_count = dict()
    freq_obj_rel_counts_per_rank_count = dict()
    freq_sub_counts_per_rank_count = dict()
    freq_obj_counts_per_rank_count = dict()

    for i, pred in enumerate(predictions):
        # print(pred)
        s, r, o, time, mode, rank = tuple(pred)
        if mode == 's':
            sub_repeat_freq(s, r, time, rank, freq_sub_rel_counts, freq_sub_counts)
        else:
            obj_repeat_freq(r, o, time, rank, freq_obj_rel_counts, freq_obj_counts)
    if binning:
        freq_sub_rel_counts = group_ranks(freq_sub_rel_counts)
        freq_obj_rel_counts = group_ranks(freq_obj_rel_counts)
        freq_sub_counts = group_ranks(freq_sub_counts)
        freq_obj_counts = group_ranks(freq_obj_counts)

    for freq_ranks, freq_metric, freq_rank_count in zip(
            [freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts, freq_obj_counts],
            [freq_sub_rel_counts_metric, freq_obj_rel_counts_metric, freq_sub_counts_metric,
             freq_obj_counts_metric],
            [freq_sub_rel_counts_per_rank_count, freq_obj_rel_counts_per_rank_count, freq_sub_counts_per_rank_count,
             freq_obj_counts_per_rank_count]):
        for freq, ranks in freq_ranks.items():
            ranks = np.array(ranks)
            # freq_metric[freq] = np.mean(ranks <= 1)
            # freq_metric[freq] = np.mean(ranks <= 3)
            freq_metric[freq] = np.mean(ranks <= 10)
            # freq_metric[freq] = np.mean(1 / ranks)
            freq_rank_count[freq] = len(ranks)
    return freq_sub_rel_counts_metric, freq_sub_counts_metric, freq_obj_rel_counts_metric, freq_obj_counts_metric, freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts, freq_obj_counts


def pred_metric_per_freq(predictions):
    freq_triple_counts = defaultdict(list)
    freq_ent_pair_counts = defaultdict(list)
    freq_sub_rel_counts = defaultdict(list)
    freq_obj_rel_counts = defaultdict(list)
    freq_sub_counts = defaultdict(list)
    freq_obj_counts = defaultdict(list)
    freq_rel_counts = defaultdict(list)

    freq_triple_counts_metric = dict()
    freq_ent_pair_counts_metric = dict()
    freq_sub_rel_counts_metric = dict()
    freq_obj_rel_counts_metric = dict()
    freq_sub_counts_metric = dict()
    freq_obj_counts_metric = dict()
    freq_rel_counts_metric = dict()

    freq_triple_counts_per_rank_count = dict()
    freq_ent_pair_counts_per_rank_count = dict()
    freq_sub_rel_counts_per_rank_count = dict()
    freq_obj_rel_counts_per_rank_count = dict()
    freq_sub_counts_per_rank_count = dict()
    freq_obj_counts_per_rank_count = dict()
    freq_rel_counts_per_rank_count = dict()

    for i, pred in enumerate(predictions):
        # print(pred)
        s, r, o, time, mode, rank = tuple(pred)
        if mode == 's':
            method = obj_metric_freq
            method(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_obj_rel_counts,
                   freq_obj_counts, freq_rel_counts)
        else:
            method = sub_metric_freq
            method(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts,
                   freq_sub_counts, freq_rel_counts)

    if binning:
        freq_triple_counts = group_ranks(freq_triple_counts)
        freq_ent_pair_counts = group_ranks(freq_ent_pair_counts)
        freq_sub_rel_counts = group_ranks(freq_sub_rel_counts)
        freq_obj_rel_counts = group_ranks(freq_obj_rel_counts)
        freq_sub_counts = group_ranks(freq_sub_counts)
        freq_obj_counts = group_ranks(freq_obj_counts)
        freq_rel_counts = group_ranks(freq_rel_counts)

    for freq_ranks, freq_metric, freq_rank_count in zip(
            [freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts,
             freq_obj_counts, freq_rel_counts],
            [freq_triple_counts_metric, freq_ent_pair_counts_metric, freq_sub_rel_counts_metric,
             freq_obj_rel_counts_metric, freq_sub_counts_metric, freq_obj_counts_metric, freq_rel_counts_metric],
            [freq_triple_counts_per_rank_count, freq_ent_pair_counts_per_rank_count,
             freq_sub_rel_counts_per_rank_count, freq_obj_rel_counts_per_rank_count,
             freq_sub_counts_per_rank_count, freq_obj_counts_per_rank_count, freq_rel_counts_per_rank_count]
    ):
        for freq, ranks in freq_ranks.items():
            ranks = np.array(ranks)
            # freq_metric[freq] = np.mean(ranks <= 1)
            # freq_metric[freq] = np.mean(ranks <= 3)
            freq_metric[freq] = np.mean(ranks <= 10)
            # freq_metric[freq] = np.mean(1 / ranks)
            freq_rank_count[freq] = len(ranks)

    return freq_triple_counts_metric, freq_ent_pair_counts_metric, freq_sub_rel_counts_metric, freq_obj_rel_counts_metric, freq_sub_counts_metric, freq_obj_counts_metric, freq_rel_counts_metric, \
           freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts, freq_obj_counts, freq_rel_counts


def plot_all(legends, linestyles):
    countss = [freq_sub_rel_counts_sub, freq_sub_counts_sub, freq_obj_rel_counts_obj, freq_obj_counts_obj, freq_triple_counts, freq_ent_pair_counts, freq_rel_counts, freq_sub_rel_counts,
                    freq_sub_counts, freq_obj_rel_counts, freq_obj_counts]
    metricss = [freq_sub_rel_sub_counts_metrics, freq_sub_counts_sub_metrics, freq_obj_rel_obj_counts_metrics, freq_obj_counts_obj_metrics, freq_triple_counts_metrics,
                    freq_ent_pair_counts_metrics, freq_rel_counts_metrics, freq_sub_rel_counts_metrics, freq_sub_counts_metrics, freq_obj_rel_counts_metrics, freq_obj_counts_metrics]
    # [freq_triple_counts_metrics, freq_ent_pair_counts_metrics, freq_rel_counts_metric, freq_sub_rel_counts_metric, freq_sub_counts_metric, freq_obj_rel_counts_metric, freq_obj_counts_metric]
    xlabels = ["subject relation", "subject", "object relation", "object", "triples", "entity pairs", "relation", "subject relation", "subject", "object relation", "object"]
    ylabels = ["subject", "subject", "object", "object", "all", "all", "all", "object", "object", "subject", "subject"]
    for metrics, xlabel, ylabel, counts in zip(metricss, xlabels, ylabels, countss):
        plot_metrics(metrics, xlabel, ylabel, legends, counts, linestyles)


def plot_metrics(metrics, xlabel, ylabel, legends, counts, linestyles):
    plt.figure(figsize=(3, 9/4), dpi=400, facecolor='w', edgecolor='k')
    # plt.figure(figsize=(8, 6), dpi=400, facecolor='w', edgecolor='k')
    # plt.figure(dpi=200, facecolor='w', edgecolor='k')
    # if xlabel == 'relation': return
    # if xlabel in ['relation', "subject", "object"]: return
    # window_length = 29
    # if "icews14" in dataset and xlabel == 'triples':
    #     window_length = 9
    # if "gdelt" in dataset:
    #     window_length += 10
    size = int(len(metrics[0]) / 1.5)
    window_length = size - 1 if size % 2 == 0 else size
    for metric, legend, count, linestyle in zip(metrics, legends, counts, linestyles):
        metric = {k: v for k, v in sorted(metric.items(), key=lambda item: item[0])}
        x = np.array(list(metric.keys()))
        y = np.array(list(metric.values()))
        log_x = np.log(np.array([x + 0.1 if x == 0 else x for x in list(metric.keys())]))
        log_y = np.log(np.array([x + 0.1 if x == 0 else x for x in list(metric.values())]))
        log_y_hat = savgol_filter(log_y, window_length, 3)
        y_hat = np.exp(log_y_hat)

        # markevery = []
        # int2idx = {}
        # for n, val in enumerate(log_x):
        #     if val % 1 < 0.1:
        #         int(round(val))
        #         markevery.append(n)
        marker = 'o' if linestyle == '' else ''
        linestyle = '-' if linestyle == '' else linestyle
        # print(np.max(x))
        # pdb.set_trace()
        plt.plot(log_x, y_hat, label=legend, ms=2, linestyle=linestyle, marker=marker)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
    plt.xlabel("log of {} frequency".format(xlabel))
    plt.ylabel("{} query Hits@10".format(ylabel))
    plt.savefig(os.path.join(exp_folder, "{}_freq_{}_query_{}_graphs_hit@10.png".format(xlabel, ylabel, train_seq_len)), bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':
    args = get_args()

    paths = [args.path_1, args.path_2, args.path_3, args.path_4]

    # paths = paths[:1]
    binning = False
    prediction_file = paths[0]
    # pdb.set_trace()
    splitted = prediction_file.split("-")
    dataset = splitted[1] if "05-15" not in prediction_file \
        else "{}-{}".format(splitted[1], splitted[2])
    # pdb.set_trace()
    bidirectional = args.bidirectional
    if bidirectional:
        train_seq_len = 8
    else:
        train_seq_len = 15

    dataset = os.path.join("interpolation", dataset)

    train_data, train_times = load_quadruples(dataset, 'train.txt')
    max_time_step = len(train_times)
    num_ents, num_rels = get_total_number(dataset, 'stat.txt')
    id2ent, id2rel = id2entrel(dataset, num_rels)
    # print(dataset)
    print(prediction_file)
    sub_rel_to_ob, obj_rel_to_sub, sub_to_ob, ob_to_sub, rel_to_ob, rel_to_sub = construct_ref_data(train_data)

    triple_freq_per_time_step, ent_pair_freq_per_time_step, sub_freq_per_time_step, obj_freq_per_time_step, \
    rel_freq_per_time_step, sub_rel_freq_per_time_step, obj_rel_freq_per_time_step = count_entity_freq_per_train_graph(train_data)

    freq_sub_rel_sub_counts_metrics = []
    freq_sub_counts_sub_metrics = []
    freq_obj_rel_obj_counts_metrics = []
    freq_obj_counts_obj_metrics = []
    freq_triple_counts_metrics = []
    freq_ent_pair_counts_metrics = []
    freq_sub_rel_counts_metrics = []
    freq_obj_rel_counts_metrics = []
    freq_sub_counts_metrics = []
    freq_obj_counts_metrics = []
    freq_rel_counts_metrics = []

    freq_triple_counts = []
    freq_ent_pair_counts = []
    freq_sub_rel_counts = []
    freq_obj_rel_counts = []
    freq_sub_counts = []
    freq_obj_counts = []
    freq_rel_counts = []
    freq_sub_rel_counts_sub = []
    freq_obj_rel_counts_obj = []
    freq_sub_counts_sub = []
    freq_obj_counts_obj = []

    for path in paths:
        with open(path, 'rb') as filehandle:
            predictions = pickle.load(filehandle)
        freq_sub_rel_sub_count_metric, freq_sub_count_sub_metric, freq_obj_rel_obj_count_metric, freq_obj_count_obj_metric, freq_sub_rel_count_sub, freq_obj_rel_count_obj, \
            freq_sub_count_sub, freq_obj_count_obj = calc_hit_10_per_score(predictions)

        freq_triple_counts_metric, freq_ent_pair_counts_metric, freq_sub_rel_counts_metric, freq_obj_rel_counts_metric, freq_sub_counts_metric, freq_obj_counts_metric, freq_rel_counts_metric, \
            freq_triple_count, freq_ent_pair_count, freq_sub_rel_count, freq_obj_rel_count, freq_sub_count, freq_obj_count, freq_rel_count = pred_metric_per_freq(predictions)
        for metrics, metric in zip(
            [freq_sub_rel_sub_counts_metrics, freq_sub_counts_sub_metrics, freq_obj_rel_obj_counts_metrics, freq_obj_counts_obj_metrics, freq_triple_counts_metrics,
             freq_ent_pair_counts_metrics, freq_sub_rel_counts_metrics, freq_obj_rel_counts_metrics, freq_sub_counts_metrics, freq_obj_counts_metrics, freq_rel_counts_metrics],
            [freq_sub_rel_sub_count_metric, freq_sub_count_sub_metric, freq_obj_rel_obj_count_metric, freq_obj_count_obj_metric, freq_triple_counts_metric,
             freq_ent_pair_counts_metric, freq_sub_rel_counts_metric, freq_obj_rel_counts_metric, freq_sub_counts_metric, freq_obj_counts_metric, freq_rel_counts_metric]
        ):
            metrics.append(metric)

        for counts, count in zip(
            [freq_sub_rel_counts_sub, freq_obj_rel_counts_obj, freq_sub_counts_sub, freq_obj_counts_obj, freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts, freq_obj_counts, freq_rel_counts],
            [freq_sub_rel_count_sub, freq_obj_rel_count_obj, freq_sub_count_sub, freq_obj_count_obj, freq_triple_count, freq_ent_pair_count, freq_sub_rel_count, freq_obj_rel_count, freq_sub_count, freq_obj_count, freq_rel_count]
        ):
            counts.append(count)
    version = 1
    exp_folder = "analysis/{}_aggregated_figure_version_{}".format(dataset, version)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    # plot_all(["GRRGCN-ensemble", "BiGRRGCN-ensemble", "GRRGCN", "BiGRRGCN", "SRGCN", "DE"])
    # plot_all(["TeMP-GRU-Ensemble", "TeMP-GRU-Gating", "TeMP-GRU-Vanilla", "SRGCN", "DE"], ['s', 'o', 'v', '^', '.'])
    # plot_all(["TeMP-GRU-Ensemble", "TeMP-GRU-Gating", "TeMP-GRU-Vanilla", "SRGCN", "DE"], [':', '-', '-.', '--', ''])
    plot_all(["TeMP-GRU-Gating", "TeMP-GRU-Vanilla", "SRGCN", "DE"], ['-', '-.', '--', ''])