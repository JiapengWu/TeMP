from utils.dataset import *
from utils.args import process_args
import pickle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pdb
import collections
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from utils.frequency import construct_ref_data, get_history_within_distance, count_entity_freq_per_train_graph

def get_batch_graph_list(t_list, seq_len, graph_dict):
    times = list(graph_dict.keys())
    # time_unit = times[1] - times[0]  # compute time unit
    time_list = []

    t_list = t_list.sort(descending=True)[0]
    g_list = []
    # s_lst = [t/15 for t in times]
    # print(s_lst)
    for tim in t_list:
        # length = int(tim / time_unit) + 1
        # cur_seq_len = seq_len if seq_len <= length else length
        length = times.index(tim) + 1
        time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
        time_list.append(([None] * (seq_len - len(time_seq))) + time_seq)
        g_list.append(([None] * (seq_len - len(time_seq))) + [graph_dict[t] for t in time_seq])
    t_batched_list = [list(x) for x in zip(*time_list)]
    g_batched_list = [list(x) for x in zip(*g_list)]
    return g_batched_list, t_batched_list

def get_train_data_freq():
    # graph_dict_train, _, _ = build_interpolation_graphs(args)
    train_entity_path = 'analysis/{}_train_entity.pt'.format(dataset)
    train_ent_pair_path = 'analysis/{}_train_ent_pair.pt'.format(dataset)
    train_ent_rel_path = 'analysis/{}_train_ent_rel.pt'.format(dataset)
    entity_freq_per_time_step_path = 'analysis/{}_entity_freq_per_time.pt'.format(dataset)
    # if os.path.isfile(train_entity_path) and os.path.isfile(train_ent_pair_path) \
    #         and os.path.isfile(train_ent_rel_path) and os.path.isfile(entity_freq_per_time_step_path):
    #     with open(train_entity_path, 'rb') as filehandle:
    #         train_entity_lst = pickle.load(filehandle)
    #     with open(train_ent_pair_path, 'rb') as filehandle:
    #         train_ent_pair_lst = pickle.load(filehandle)
    #     with open(train_ent_rel_path, 'rb') as filehandle:
    #         train_ent_rel_lst = pickle.load(filehandle)
    #     with open(entity_freq_per_time_step_path, 'rb') as filehandle:
    #         entity_freq_per_time_step = pickle.load(filehandle)
    # else:
    train_graph_dict, _, _ = build_interpolation_graphs(args)
    train_entity_lst = []
    train_ent_rel_lst = []
    train_ent_pair_lst = []
    for quad in train_data:
        s, r, o, t = tuple(quad)
        # train_entity_lst.extend([s, o])
        train_entity_lst.extend([s])
        train_ent_pair_lst.extend([(s, o)])
        train_ent_rel_lst.extend([(s, r), (o, r)])

            # pdb.set_trace()
        # with open(train_entity_path, 'wb') as filehandle:
        #     pickle.dump(train_entity_lst, filehandle)
        # with open(train_ent_pair_path, 'wb') as filehandle:
        #     pickle.dump(train_ent_pair_lst, filehandle)
        # with open(train_ent_rel_path, 'wb') as filehandle:
        #     pickle.dump(train_ent_rel_lst, filehandle)
        # with open(entity_freq_per_time_step_path, 'wb') as filehandle:
        #     pickle.dump(entity_freq_per_time_step, filehandle)
    return Counter(train_entity_lst), Counter(train_ent_rel_lst), Counter(train_ent_pair_lst)


def calc_per_entity_prediction(predictions):
    per_entity_ranks = defaultdict(list)
    per_entity_pair_ranks = defaultdict(list)
    per_entity_rel_ranks = defaultdict(list)
    for pred in predictions:
        s, r, o, time, mode, rank = tuple(pred)
        # pdb.set_trace()
        per_entity_pair_ranks[(s, o)].append(rank)
        if mode == 's':
            per_entity_ranks[o].append(rank)
            per_entity_rel_ranks[(o, r)].append(rank)
        else:
            per_entity_ranks[s].append(rank)
            per_entity_rel_ranks[(s, r)].append(rank)
    return per_entity_ranks, per_entity_pair_ranks, per_entity_rel_ranks


def hist_freq_entity_rel_mrr(train_ent_rel_freq, per_entity_rel_ranks):
    freq_ranks = defaultdict(list)
    freq_mrr = {}
    for entity_rel, freq in train_ent_rel_freq.items():
        # pdb.set_trace()
        ranks = per_entity_rel_ranks[entity_rel]
        freq_ranks[freq].extend(ranks)

    for freq, ranks in freq_ranks.items():
        freq_mrr[freq] = np.mean(1 / np.array(ranks))

    plt.scatter(list(freq_mrr.keys()), list(freq_mrr.values()), s=marker_size)
    plt.xlabel("frequency of entity relation")
    plt.ylabel("mrr")
    plt.savefig(os.path.join(exp_folder, "entity_rel_freq_mrr.png"))
    plt.clf()
    return freq_mrr


def hist_freq_entity_pair_mrr(train_ent_pair_freq, per_entity_pair_ranks):
    freq_ranks = defaultdict(list)
    freq_mrr = {}
    counted = defaultdict(bool)
    for entity_pair in train_ent_pair_freq.keys():
        # pdb.set_trace()
        if not counted[entity_pair]:
            s, o = entity_pair
            counted[(s, o)] = counted[(o, s)] = True
            freq = train_ent_pair_freq[(s, o)] + train_ent_pair_freq[(o, s)]  # sum of freqs, guaranteed to be integer
            ranks = per_entity_pair_ranks[(s, o)] + per_entity_pair_ranks[(o, s)]  # list concat

            freq_ranks[freq].extend(ranks)

    for freq, ranks in freq_ranks.items():
        freq_mrr[freq] = np.mean(1 / np.array(ranks))

    plt.scatter(list(freq_mrr.keys()), list(freq_mrr.values()), s=marker_size)
    plt.xlabel("frequency of entity pairs")
    plt.ylabel("mrr")
    plt.savefig(os.path.join(exp_folder, "entity_pair_freq_mrr.png"))
    plt.clf()


def hist_freq_entity_mrr(train_entity_freq, per_entity_ranks):
    freq_ranks = defaultdict(list)
    freq_mrr = {}
    for entity, freq in train_entity_freq.items():
        ranks = per_entity_ranks[entity]
        freq_ranks[freq].extend(ranks)

    for freq, ranks in freq_ranks.items():
        freq_mrr[freq] = np.mean(1 / np.array(ranks))
    # print(freq_mrr)
    plt.scatter(list(freq_mrr.keys()), list(freq_mrr.values()), s=marker_size)
    plt.xlabel("frequency of entity")
    plt.ylabel("mrr")
    plt.savefig(os.path.join(exp_folder, "entity_freq_mrr.png"))
    plt.clf()


def calc_metrics_per_time(ranks):
    mrrs = defaultdict(int)
    hit_1s = defaultdict(int)
    hit_3s = defaultdict(int)
    hit_10s = defaultdict(int)
    for time in ranks.keys():
        rank_at_t = np.array(ranks[time])
        mrrs[time] = np.mean(1 / rank_at_t)
        hit_1s[time] = np.mean((rank_at_t <= 1))
        hit_3s[time] = np.mean((rank_at_t <= 3))
        hit_10s[time] = np.mean((rank_at_t <= 10))
    return mrrs, hit_1s, hit_3s, hit_10s


def obj_metric(s, r, o, time, rank, triple_counts, ent_pair_counts, obj_rel_counts, obj_counts, rel_counts, rank_sub):
    triple_count = 0
    obj_count = 0
    rel_count = 0
    obj_rel_count = 0
    ent_pair_count = 0
    for t in range(max(0, time - train_seq_len + 1), time + 1):
        triple_count += triple_freq_per_time_step[(s, r, o)][t]
        ent_pair_count += ent_pair_freq_per_time_step[(s, o)][t]
        obj_rel_count += obj_rel_freq_per_time_step[(o, r)][t]
        obj_count += obj_freq_per_time_step[o][t]
        rel_count += rel_freq_per_time_step[r][t]
    triple_counts[time] += triple_count
    ent_pair_counts[time] += ent_pair_count
    obj_rel_counts[time] += obj_rel_count
    obj_counts[time] += obj_count
    rel_counts[time] += rel_count
    rank_sub[time].append(rank)


def sub_metric(s, r, o, time, rank, triple_counts, ent_pair_counts, sub_rel_counts, sub_counts, rel_counts, rank_obj):
    triple_count = 0
    sub_count = 0
    rel_count = 0
    sub_rel_count = 0
    ent_pair_count = 0
    for t in range(max(0, time - train_seq_len + 1), time + 1):
        triple_count += triple_freq_per_time_step[(s, r, o)][t]
        ent_pair_count += ent_pair_freq_per_time_step[(s, o)][t]
        sub_rel_count += sub_rel_freq_per_time_step[(s, r)][t]
        sub_count += sub_freq_per_time_step[s][t]
        rel_count += rel_freq_per_time_step[r][t]
    triple_counts[time] += triple_count
    ent_pair_counts[time] += ent_pair_count
    sub_rel_counts[time] += sub_rel_count
    sub_counts[time] += sub_count
    rel_counts[time] += rel_count
    rank_obj[time].append(rank)


def pred_metric_per_time(predictions):
    triple_counts = defaultdict(int)
    ent_pair_counts = defaultdict(int)
    sub_rel_counts = defaultdict(int)
    obj_rel_counts = defaultdict(int)
    sub_counts = defaultdict(int)
    obj_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    rank_sub = defaultdict(list)
    rank_obj = defaultdict(list)

    for i, pred in enumerate(predictions):
        # print(pred)
        s, r, o, time, mode, rank = tuple(pred)
        if mode == 's':
            obj_metric(s, r, o, time, rank, triple_counts, ent_pair_counts, obj_rel_counts, obj_counts, rel_counts,
                       rank_sub)
        else:
            sub_metric(s, r, o, time, rank, triple_counts, ent_pair_counts, sub_rel_counts, sub_counts, rel_counts,
                       rank_obj)

    triple_counts = sort_dict(triple_counts)
    ent_pair_counts = sort_dict(ent_pair_counts)
    sub_rel_counts = sort_dict(sub_rel_counts)
    obj_rel_counts = sort_dict(obj_rel_counts)
    sub_counts = sort_dict(sub_counts)
    obj_counts = sort_dict(obj_counts)
    rel_counts = sort_dict(rel_counts)
    rank_sub = sort_dict(rank_sub)
    rank_obj = sort_dict(rank_obj)
    mrrs_sub, hit_1s_sub, hit_3s_sub, hit_10s_sub = calc_metrics_per_time(rank_sub)
    mrrs_obj, hit_1s_obj, hit_3s_obj, hit_10s_obj = calc_metrics_per_time(rank_obj)

    # num_edges_per_time = np.array([len(ranks) for time, ranks in rank_sub.items()])
    # plt.plot(list(mrrs_obj.keys()), np.array(list(mrrs_obj.values())) * 0.16)
    # plt.plot(list(triple_counts.keys()), np.array(list(mrrs_sub.values())) / num_edges_per_time)
    # plt.show()
    # plt.plot(list(mrrs_obj.keys()), np.array(list(mrrs_obj.values()))*17)
    # plt.plot(list(ent_pair_counts.keys()), np.array(list(ent_pair_counts.values())) / num_edges_per_time)
    # plt.show()
    # plt.plot(list(mrrs_obj.keys()), np.array(list(mrrs_obj.values())) * 7)
    # plt.plot(list(sub_rel_counts.keys()), np.array(list(sub_rel_counts.values())) / num_edges_per_time)
    # plt.show()
    # plt.plot(list(mrrs_obj.keys()), np.array(list(mrrs_obj.values())) * 50)
    # plt.plot(list(sub_counts.keys()), np.array(list(sub_counts.values())) / num_edges_per_time)
    # plt.show()
    # plt.plot(list(mrrs_obj.keys()), np.array(list(mrrs_obj.values())) * 600)
    # plt.plot(list(rel_counts.keys()), np.array(list(rel_counts.values())) / num_edges_per_time)
    # plt.show()
    # plt.clf()


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

    # if obj_rel_count <= 10 and obj_rel_count > 0:
    if False:
        s_string = id2ent[s]
        o_string = id2ent[o]
        rel_string = id2rel[r]
        print("At time {}, subject rank {}, triple: {}\t{}\t{}".format(time, rank, s_string, rel_string, o_string))
        sub_lst_all = []
        for t in range(max(0, time - train_seq_len + 1), time + 1):
            sub_lst = obj_rel_to_sub[(o, r)][t]
            if len(sub_lst) > 0:
                sub_lst_all.extend(sub_lst)
                print("At time {}".format(t))
                for sub in sub_lst:
                    print("{}\t{}\t{}".format(id2ent[sub], rel_string, o_string))
        print("Answer {}in the history set".format("" if s in sub_lst_all else "not "))
        if s in sub_lst_all:
            pdb.set_trace()

    '''
    if obj_count <= 10 and obj_count > 0:
        s_string = id2ent[s]
        o_string = id2ent[o]
        rel_string = id2rel[r]
        print("At time {}, subject rank {}, triple: {}\t{}\t{}".format(time, rank, s_string, rel_string, o_string))
        sub_lst_all = []
        for t in range(max(0, time - train_seq_len + 1), time + 1):
            sub_lst = ob_to_sub[o][t]
            if len(sub_lst) > 0:
                sub_lst_all.extend(sub_lst)
                print("At time {}".format(t))
                for sub in sub_lst:
                    print("{}\t{}".format(id2ent[sub], o_string))
        print("Answer {}in the history set".format("" if s in sub_lst_all else "not "))
        pdb.set_trace()
        '''


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

    # if sub_rel_count <= 10 and sub_rel_count > 0:
    if False:
        s_string = id2ent[s]
        o_string = id2ent[o]
        rel_string = id2rel[r]
        print("At time {}, object rank {}, triple: {}\t{}\t{}".format(time, rank, s_string, rel_string, o_string))
        obj_lst_all = []
        for t in range(max(0, time - train_seq_len + 1), time + 1):
            obj_lst = sub_rel_to_ob[(s, r)][t]
            if len(obj_lst) > 0:
                obj_lst_all.extend(obj_lst)
                print("At time {}".format(t))
                for obj in obj_lst:
                    print("{}\t{}\t{}".format(o_string, rel_string, id2ent[obj]))
        print("Answer {}in the history set".format("" if s in obj_lst_all else "not "))
        pdb.set_trace()


def obj_metric_freq_all(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_obj_rel_counts, freq_obj_counts, freq_rel_counts):
    triple_count = np.sum(list(triple_freq_per_time_step[(s, r, o)].values()))
    ent_pair_count = np.sum(list(ent_pair_freq_per_time_step[(s, o)].values()))
    obj_rel_count = np.sum(list(obj_rel_freq_per_time_step[(o, r)].values()))
    obj_count = np.sum(list(obj_freq_per_time_step[o].values()))
    rel_count = np.sum(list(rel_freq_per_time_step[r].values()))

    freq_triple_counts[triple_count].append(rank)
    freq_ent_pair_counts[ent_pair_count].append(rank)
    freq_obj_rel_counts[obj_rel_count].append(rank)
    freq_obj_counts[obj_count].append(rank)
    freq_rel_counts[rel_count].append(rank)


def sub_metric_freq_all(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_sub_counts, freq_rel_counts):
    triple_count = np.sum(list(triple_freq_per_time_step[(s, r, o)].values()))
    ent_pair_count = np.sum(list(ent_pair_freq_per_time_step[(s, o)].values()))
    sub_rel_count = np.sum(list(sub_rel_freq_per_time_step[(s, r)].values()))
    sub_count = np.sum(list(sub_freq_per_time_step[s].values()))
    rel_count = np.sum(list(rel_freq_per_time_step[r].values()))
    # pdb.set_trace()

    freq_triple_counts[triple_count].append(rank)
    freq_ent_pair_counts[ent_pair_count].append(rank)
    freq_sub_rel_counts[sub_rel_count].append(rank)
    freq_sub_counts[sub_count].append(rank)
    freq_rel_counts[rel_count].append(rank)


def sort_dict(dictionary):
    return collections.OrderedDict(sorted(dictionary.items()))


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
            method = obj_metric_freq_all if all else obj_metric_freq
            method(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_obj_rel_counts, freq_obj_counts, freq_rel_counts)
        else:
            method = sub_metric_freq_all if all else sub_metric_freq
            method(s, r, o, time, rank, freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_sub_counts, freq_rel_counts)

    for freq_ranks, freq_metric, freq_rank_count in zip([freq_triple_counts, freq_ent_pair_counts, freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts, freq_obj_counts, freq_rel_counts],
                                       [freq_triple_counts_metric, freq_ent_pair_counts_metric, freq_sub_rel_counts_metric, freq_obj_rel_counts_metric, freq_sub_counts_metric, freq_obj_counts_metric, freq_rel_counts_metric],
                                       [freq_triple_counts_per_rank_count, freq_ent_pair_counts_per_rank_count, freq_sub_rel_counts_per_rank_count, freq_obj_rel_counts_per_rank_count,
                                        freq_sub_counts_per_rank_count, freq_obj_counts_per_rank_count, freq_rel_counts_per_rank_count]):
        for freq, ranks in freq_ranks.items():
            ranks = np.array(ranks)
            # freq_metric[freq] = np.mean(ranks <= 1)
            # freq_metric[freq] = np.mean(ranks <= 3)
            freq_metric[freq] = np.mean(ranks <= 10)
            # freq_metric[freq] = np.mean(1 / ranks)
            freq_rank_count[freq] = len(ranks)

    plot_metric_per_freq(freq_triple_counts_metric, freq_triple_counts_per_rank_count, "triples", "overall")
    plot_metric_per_freq(freq_ent_pair_counts_metric, freq_ent_pair_counts_per_rank_count, "entity pairs", "overall")
    plot_metric_per_freq(freq_rel_counts_metric, freq_rel_counts_per_rank_count, "relation", "overall")
    plot_metric_per_freq(freq_sub_rel_counts_metric, freq_sub_rel_counts_per_rank_count, "subject relation", "object")
    plot_metric_per_freq(freq_sub_counts_metric, freq_sub_counts_per_rank_count, "subject", "object")
    plot_metric_per_freq(freq_obj_rel_counts_metric, freq_obj_rel_counts_per_rank_count, "object relation", "subject")
    plot_metric_per_freq(freq_obj_counts_metric, freq_obj_counts_per_rank_count, "object", "subject")


def plot_metric_per_freq(metric, rank_count, xlabel, ylabel):
    # list(metric.keys()) + 0.1
    x = np.log(np.array(list(metric.keys())) + 0.01)
    # pdb.set_trace()
    y = np.array(list(metric.values()))
    regressor = LinearRegression()
    regressor.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    # print(regressor.intercept_); print(regressor.coef_)
    y_fit = regressor.predict(x.reshape(-1, 1))
    plt.scatter(x, y, s=np.sqrt(np.array(list(rank_count.values()))))
    plt.plot(x, y_fit, color='red')
    plt.ylim(0, 1.05)

    # plt.xlabel("frequency of {} in last {} graphs".format(xlabel, "all" if all else train_seq_len))
    plt.ylabel("{} hit@10".format(ylabel))

    plt.savefig(os.path.join(exp_folder, "{}_freq_{}_query_{}_graphs_hit_10.png".format(xlabel, ylabel, "all" if all else train_seq_len)))
    plt.clf()
    # plt.show()


def per_graph_freq_entity_mrr(entity_freq_per_time_step, predictions):
    freq_ranks = defaultdict(list)
    freq_mrr = {}
    for pred in predictions:
        s, r, o, time, mode, rank = tuple(pred)
        entity = o if mode == 's' else s
        freq = 0
        for t in range(max(0, time - train_seq_len + 1), time + 1):
            try:
                freq += entity_freq_per_time_step[t][entity]
            except:
                pdb.set_trace()
        freq_ranks[freq].append(rank)

    for freq, ranks in freq_ranks.items():
        freq_mrr[freq] = np.mean(1 / np.array(ranks))

    plt.scatter(list(freq_mrr.keys()), list(freq_mrr.values()), s=marker_size)
    plt.xlabel("frequency of entity in last {} graphs".format(train_seq_len))
    plt.ylabel("mrr")
    plt.savefig(os.path.join(exp_folder, "entity_freq_last_n_graph_mrr.png"))
    plt.clf()


def calc_metrics(predictions):
    ranks = np.array([pred[-1] for pred in predictions])
    mrr = np.mean(1 / ranks)
    hit_1 = np.mean((ranks <= 1))
    hit_3 = np.mean((ranks <= 3))
    hit_10 = np.mean((ranks <= 10))
    print("MRR: {}".format(mrr))
    print("HIT@10: {}".format(hit_10))
    print("HIT@3: {}".format(hit_3))
    print("HIT@1: {}".format(hit_1))



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

    for freq_ranks, freq_metric, freq_rank_count in zip([freq_sub_rel_counts, freq_obj_rel_counts, freq_sub_counts, freq_obj_counts],
                                       [freq_sub_rel_counts_metric, freq_obj_rel_counts_metric, freq_sub_counts_metric, freq_obj_counts_metric],
                                       [freq_sub_rel_counts_per_rank_count, freq_obj_rel_counts_per_rank_count, freq_sub_counts_per_rank_count, freq_obj_counts_per_rank_count]):
        for freq, ranks in freq_ranks.items():
            ranks = np.array(ranks)
            # freq_metric[freq] = np.mean(ranks <= 1)
            # freq_metric[freq] = np.mean(ranks <= 3)
            freq_metric[freq] = np.mean(ranks <= 10)
            # freq_metric[freq] = np.mean(1 / ranks)
            freq_rank_count[freq] = len(ranks)

    plot_metric_per_freq(freq_sub_rel_counts_metric, freq_sub_rel_counts_per_rank_count, "subject relation", "subject")
    plot_metric_per_freq(freq_sub_counts_metric, freq_sub_counts_per_rank_count, "subject", "subject")
    plot_metric_per_freq(freq_obj_rel_counts_metric, freq_obj_rel_counts_per_rank_count, "object relation", "object")
    plot_metric_per_freq(freq_obj_counts_metric, freq_obj_counts_per_rank_count, "object", "object")


def calc_mrr_per_score():
    score2ranks_o = defaultdict(list)
    score2ranks_s = defaultdict(list)
    # print(len(predictions))
    for i, pred in enumerate(predictions):
        # print(pred)
        s, r, o, time, mode, rank = tuple(pred)
        sub_rel_hist = sub_rel_to_ob[(s, r)]
        sub_obj_hist = sub_to_ob[s]
        rel_obj_hist = rel_to_ob[r]

        obj_rel_hist = obj_rel_to_sub[(o, r)]
        obj_sub_hist = ob_to_sub[o]
        rel_sub_hist = rel_to_sub[r]
        one = two = three = False
        if not mode == 's':

            if o in get_history_within_distance(sub_rel_hist, args.train_seq_len, time, args.future):
                score2ranks_o[1].append(rank)
                one = True
            if o in get_history_within_distance(sub_obj_hist, args.train_seq_len, time, args.future):
                score2ranks_o[2].append(rank)
                if not one:
                    score2ranks_o[5].append(rank)
                two = True
            if o in get_history_within_distance(rel_obj_hist, args.train_seq_len, time, args.future):
                score2ranks_o[3].append(rank)
                if not one:
                    score2ranks_o[6].append(rank)
                three = True
            if not one and not two and not three:
                score2ranks_o[4].append(rank)
        else:
            if s in get_history_within_distance(obj_rel_hist, args.train_seq_len, time, args.future):
                score2ranks_s[1].append(rank)
                one = True
            if s in get_history_within_distance(obj_sub_hist, args.train_seq_len, time, args.future):
                score2ranks_s[2].append(rank)
                if not one:
                    score2ranks_s[5].append(rank)
                two = True
            if s in get_history_within_distance(rel_sub_hist, args.train_seq_len, time, args.future):
                score2ranks_s[3].append(rank)
                if not one:
                    score2ranks_s[6].append(rank)
                three = True
            if not one and not two and not three:
                score2ranks_s[4].append(rank)

    print("object score to mrr:")
    for score, ranks in sorted(score2ranks_o.items()):
        mrr = np.mean(1 / np.array(ranks))
        print("Score: {}, mrr: {}".format(score, mrr))

    print("subject score to mrr:")
    for score, ranks in sorted(score2ranks_s.items()):
        mrr = np.mean(1 / np.array(ranks))
        print("Score: {}, mrr: {}".format(score, mrr))

def max_two_models():
    other_prediction_file = args.temporal_checkpoint
    with open(other_prediction_file, 'rb') as filehandle:
        other_predictions = pickle.load(filehandle)
    predictions_map = {}
    ranks = []
    for pred in predictions:
        predictions_map[tuple(pred[:-2])] = pred[-1]
    for pred in other_predictions:
        rank = min(pred[-1], predictions_map[tuple(pred[:-2])])
        ranks.append(rank)
    # pdb.set_trace()
    ranks = np.array(ranks)
    mrr = np.mean(1 / ranks)
    hit_1 = np.mean((ranks <= 1))
    hit_3 = np.mean((ranks <= 3))
    hit_10 = np.mean((ranks <= 10))
    print("MRR: {}".format(mrr))
    print("HIT@10: {}".format(hit_10))
    print("HIT@3: {}".format(hit_3))
    print("HIT@1: {}".format(hit_1))

def new_entitiy_metrics():
    for i, pred in enumerate(predictions):
        # print(pred)
        s, r, o, time, mode, rank = tuple(pred)


if __name__ == '__main__':
    args = process_args()
    marker_size = 5
    # prediction_file = os.path.join("analysis", args.checkpoint_path)
    # exp_folder = os.path.join("analysis", args.checkpoint_path.split(".")[0])
    prediction_file = args.checkpoint_path

    exp_folder = ".".join(prediction_file.split(".")[:-1])
    # pdb.set_trace()
    dataset = os.path.basename(args.dataset)
    splitted = args.checkpoint_path.split("-")
    dataset = splitted[1] if "05-15" not in args.checkpoint_path \
        else "{}-{}".format(splitted[1], splitted[2])
    bidirectional = True
    if bidirectional:
        train_seq_len = 8
    else:
        train_seq_len = 15

    args.dataset = os.path.join(args.dataset_dir, dataset)
    all = args.all
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    with open(prediction_file, 'rb') as filehandle:
        predictions = pickle.load(filehandle)

    train_data, train_times = load_quadruples(args.dataset, 'train.txt')
    max_time_step = len(train_times)
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    # train_graph_dict, _, _ = build_interpolation_graphs(args)
    id2ent, id2rel = id2entrel(args.dataset, num_rels)
    print(prediction_file)
    calc_metrics(predictions)
    exit()
    # max_two_models()
    sub_rel_to_ob, obj_rel_to_sub, sub_to_ob, ob_to_sub, rel_to_ob, rel_to_sub = construct_ref_data(train_data)
    train_entity_freq, train_ent_rel_freq, train_ent_pair_freq = get_train_data_freq()
    triple_freq_per_time_step, ent_pair_freq_per_time_step, sub_freq_per_time_step, obj_freq_per_time_step, \
        rel_freq_per_time_step, sub_rel_freq_per_time_step, obj_rel_freq_per_time_step = count_entity_freq_per_train_graph(train_data)

    calc_hit_10_per_score(predictions)
    pred_metric_per_freq(predictions)
    per_entity_ranks, per_entity_pair_ranks, per_entity_rel_ranks = calc_per_entity_prediction(predictions)
    # pred_metric_per_time(predictions)
    # hist_freq_entity_mrr(train_entity_freq, per_entity_ranks)
    # hist_freq_entity_pair_mrr(train_ent_pair_freq, per_entity_pair_ranks)
    # hist_freq_entity_rel_mrr(train_ent_rel_freq, per_entity_rel_ranks)
    # per_graph_freq_entity_mrr(entity_freq_per_time_step, predictions)

