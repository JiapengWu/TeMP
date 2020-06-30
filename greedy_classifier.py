# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
import json
import pdb
from utils.frequency import get_history_within_distance, count_entity_freq_per_train_graph


'''
def rank_tail(e, r):
    rank_lst = []
    # pdb.set_trace()
    try:
        rank_lst.extend(list(sub_rel_to_ob[(e, r)]))
    except:
        pass
    first_lst = list(filter(lambda x: x not in rank_lst, list(sub_to_ob[e])))
    rank_lst.extend(first_lst)
    second_lst = list(filter(lambda x: x not in rank_lst, list(rel_to_ob[r])))
    rank_lst.extend(second_lst)
    rest = np.array(list(filter(lambda x: x not in rank_lst, entities)))
    np.random.shuffle(rest)
    rank_lst.extend(rest.tolist())
    return rank_lst


def rank(s, r, o, t):
    rank_obj_lst = rank_tail(s, r)
    # pdb.set_trace()
    rank_obj = rank_obj_lst.index(o)
    rank_sub_lst = rank_tail(o, r)
    rank_sub = rank_sub_lst.index(o)
    pred_list.append([s, r, o, t, 's', rank_sub])
    pred_list.append([s, r, o, t, 'o', rank_obj])
    return [rank_obj + 1, rank_sub + 1]


def construct_ref_data():
    sub_rel_to_ob = defaultdict(set)
    sub_to_ob = defaultdict(set)
    rel_to_ob = defaultdict(set)
    for quad in train_data:
        # pdb.set_trace()
        s, r, o, t = tuple(quad)
        sub_rel_to_ob[(s, r)].add(o)
        sub_rel_to_ob[(o, r)].add(s)
        sub_to_ob[s].add(o)
        sub_to_ob[o].add(s)
        rel_to_ob[r].add(s)
        rel_to_ob[r].add(o)
    return sub_rel_to_ob, sub_to_ob, rel_to_ob
'''

'''
def rank_tail(s, r, t, true_tail):
    rank_lst = []
    if debug:
        print("Tail ranking:")
    ent_rel_hist = sub_rel_to_ob[(s, r)]
    rank_lst.extend(distance_based_ent_rel_ranks(ent_rel_hist, t))

    ent_obj_hist = sub_to_ob[s]
    rel_obj_hist = rel_to_ob[r]
    entity_ranks, rel_ranks = distance_based_ent_or_rel_ranks(ent_obj_hist, rel_obj_hist, t)
    rank_lst.extend(entity_ranks + rel_ranks + entities)
    rank_obj_lst = list(dict.fromkeys(rank_lst))
    # if debug:
    #     print("Tail ranking:")
    #     for r in rank_obj_lst[:10]:
    #         print(id2ent[r])
    #     print()
    return rank_obj_lst.index(true_tail)


def rank_head(o, r, t, true_head):
    rank_lst = []
    if debug:
        print("Head ranking:")
    ent_rel_hist = obj_rel_to_sub[(o, r)]
    rank_lst.extend(distance_based_ent_rel_ranks(ent_rel_hist, t))
    ent_sub_hist = ob_to_sub[o]
    rel_sub_hist = rel_to_sub[r]
    entity_ranks, rel_ranks = distance_based_ent_or_rel_ranks(ent_sub_hist, rel_sub_hist, t)
    rank_lst.extend(entity_ranks + rel_ranks + entities)
    rank_sub_lst = list(dict.fromkeys(rank_lst))
    # if debug:
    #     print("Head ranking:")
    #     for rank in rank_sub_lst[:10]:
    #         print(id2ent[rank])
    #     print()
    return rank_sub_lst.index(true_head)

def construct_ref_data():
    sub_rel_to_ob = defaultdict(lambda: defaultdict(list))
    obj_rel_to_sub = defaultdict(lambda: defaultdict(list))
    sub_to_ob = defaultdict(lambda: defaultdict(list))
    ob_to_sub = defaultdict(lambda: defaultdict(list))
    rel_to_ob = defaultdict(lambda: defaultdict(list))
    rel_to_sub = defaultdict(lambda: defaultdict(list))
    for quad in train_data:
        # pdb.set_trace()
        s, r, o, t = tuple(quad)
        sub_rel_to_ob[(s, r)][t].append(o)
        obj_rel_to_sub[(o, r)][t].append(s)
        sub_to_ob[s][t].append(o)
        ob_to_sub[o][t].append(s)
        rel_to_sub[r][t].append(s)
        rel_to_ob[r][t].append(o)
    return sub_rel_to_ob, obj_rel_to_sub, sub_to_ob, ob_to_sub, rel_to_ob, rel_to_sub
    

def distance_based_ent_rel_ranks(ent_rel_hist, t):
    dist2rank = defaultdict(list)
    ranks = []
    dists = []
    for time in ent_rel_hist.keys():
        if time - t == 0: continue
        dist2rank[abs(t - time)].append(ent_rel_hist[time])
    for dist in sorted(dist2rank.keys()):
        ranks.append(flatten(dist2rank[dist]))
        dists.extend([dist] * len(dist2rank[dist]))
    if debug:
        for i, (rank, dist) in enumerate(zip(flatten(ranks), dists)):
            # pdb.set_trace()
            print("{}\t{}".format(id2ent[rank], dist))
            if i >= 10: break
        print()

    # ranks = [flatten(dist2rank[dist]) for dist in sorted(dist2rank.keys())]
    return flatten(ranks)


def distance_based_ent_or_rel_ranks(ent_hist, rel_hist, t):
    dist2ent_rank = defaultdict(list)
    for time in ent_hist.keys():
        dist2ent_rank[abs(t - time)].append(ent_hist[time])
    entity_ranks = [flatten(dist2ent_rank[dist]) for dist in sorted(dist2ent_rank.keys())]

    dist2rel_rank = defaultdict(list)
    for time in rel_hist.keys():
        dist2rel_rank[abs(t - time)].append(rel_hist[time])
    rel_ranks = [flatten(dist2rel_rank[dist]) for dist in sorted(dist2rel_rank.keys())]

    return flatten(entity_ranks), flatten(rel_ranks)

'''


def construct_ref_data(train_data):
    sub_rel_to_ob = defaultdict(lambda: defaultdict(list))
    obj_rel_to_sub = defaultdict(lambda: defaultdict(list))
    sub_to_ob = defaultdict(lambda: defaultdict(list))
    ob_to_sub = defaultdict(lambda: defaultdict(list))
    rel_to_ob = defaultdict(lambda: defaultdict(list))
    rel_to_sub = defaultdict(lambda: defaultdict(list))
    for quad in train_data:
        s, r, o, t = tuple(quad)
        if vote == 'recency':
            sub_rel_to_ob[(s, r)][t].append(o)
            obj_rel_to_sub[(o, r)][t].append(s)
        else:
            sub_rel_to_ob[(s, r)][o].append(t)
            obj_rel_to_sub[(o, r)][s].append(t)
        sub_to_ob[s][t].append(o)
        ob_to_sub[o][t].append(s)
        rel_to_sub[r][t].append(s)
        rel_to_ob[r][t].append(o)
    # if 'gdelt' in args.dataset:
    #     for pattern in sub_rel_to_ob, obj_rel_to_sub:
    #         for key1 in pattern.keys():
    #             for key2 in pattern[key1].keys():
    #                 # pdb.set_trace()
    #                 pattern[key1][key2] = list(set(pattern[key1][key2]))

    return sub_rel_to_ob, obj_rel_to_sub, sub_to_ob, ob_to_sub, rel_to_ob, rel_to_sub


flatten = lambda l: [item for sublist in l for item in sublist]


def exp_decay_scoring(history, cur_time):
    entity2score = dict()
    for entity in history.keys():
        entity_time_list = np.array(history[entity])
        dist_lst = np.abs(cur_time - entity_time_list)

        non_zero_mask = dist_lst.nonzero()[0]
        dist_lst_non_zero = dist_lst[non_zero_mask]
        smoothed_sum = np.sum(np.exp(-lam * dist_lst_non_zero))
        entity2score[entity] = smoothed_sum
    return {k: v for k, v in sorted(entity2score.items(), key=lambda item: item[1], reverse=True)}


def distance_based_ent_rel_ranks(ent_rel_hist, t):
    if vote == 'recency':
        dist2rank = defaultdict(list)
        ranks = []
        dists = []
        for time in ent_rel_hist.keys():
            # if time - t == 0: continue
            time_diff = t - time
            condition = time_diff <= args.train_seq_len and time_diff >= 0 if not args.future else abs(time_diff) <= args.train_seq_len
            if condition:
                dist2rank[abs(t - time)].append(ent_rel_hist[time])
        for dist in sorted(dist2rank.keys()):
            ranks.append(flatten(dist2rank[dist]))
            dists.extend([dist] * len(dist2rank[dist]))
        if debug:
            for i, (rank, dist) in enumerate(zip(flatten(ranks), dists)):
                print("{}\t{}".format(id2ent[rank], dist))
                if i >= 10: break
            print()
        return flatten(ranks)

    else:
        entity2score_sorted = exp_decay_scoring(ent_rel_hist, t)
        if debug:
            for i, (entity, score) in enumerate(entity2score_sorted.items()):
                print("{}\t{}".format(id2ent[entity], score))
                if i >= 10: break
            print()
        return list(entity2score_sorted.keys())


def distance_based_ent_or_rel_ranks(ent_hist, rel_hist, t):
    dist2ent_rank = defaultdict(list)
    for time in ent_hist.keys():
        time_diff = t - time
        condition = time_diff <= args.train_seq_len and time_diff >= 0 if not args.future else abs(time_diff) <= args.train_seq_len
        if condition:
            dist2ent_rank[abs(t - time)].append(ent_hist[time])
    entity_ranks = [flatten(dist2ent_rank[dist]) for dist in sorted(dist2ent_rank.keys())]

    dist2rel_rank = defaultdict(list)
    for time in rel_hist.keys():
        time_diff = t - time
        condition = time_diff <= args.train_seq_len and time_diff >= 0 if not args.future else abs(time_diff) <= args.train_seq_len
        if condition:
            dist2rel_rank[abs(t - time)].append(rel_hist[time])
    rel_ranks = [flatten(dist2rel_rank[dist]) for dist in sorted(dist2rel_rank.keys())]

    return flatten(entity_ranks), flatten(rel_ranks)


def rank_tail(s, r, t, true_tail):
    rank_lst = []
    if debug:
        print("Tail ranking:")
    ent_rel_hist = sub_rel_to_ob[(s, r)]
    rank_lst.extend(distance_based_ent_rel_ranks(ent_rel_hist, t))

    ent_obj_hist = sub_to_ob[s]
    rel_obj_hist = rel_to_ob[r]
    entity_ranks, rel_ranks = distance_based_ent_or_rel_ranks(ent_obj_hist, rel_obj_hist, t)
    rank_lst.extend(entity_ranks + rel_ranks + entities)
    rank_obj_lst = list(dict.fromkeys(rank_lst))
    if filtered:
        filtered_tails = [x for x in true_tails[t][(s, r)] if x != true_tail]
        for i in filtered_tails:
            rank_obj_lst.remove(i)
        rank_obj_lst.extend(filtered_tails)

    return rank_obj_lst.index(true_tail)


def rank_head(o, r, t, true_head):
    rank_lst = []
    if debug:
        print("Head ranking:")
    ent_rel_hist = obj_rel_to_sub[(o, r)]
    rank_lst.extend(distance_based_ent_rel_ranks(ent_rel_hist, t))
    ent_sub_hist = ob_to_sub[o]
    rel_sub_hist = rel_to_sub[r]
    entity_ranks, rel_ranks = distance_based_ent_or_rel_ranks(ent_sub_hist, rel_sub_hist, t)
    rank_lst.extend(entity_ranks + rel_ranks + entities)
    rank_sub_lst = list(dict.fromkeys(rank_lst))
    if filtered:
        filtered_heads = [x for x in true_heads[t][(o, r)] if x != true_head]
        for i in filtered_heads:
            rank_sub_lst.remove(i)
        rank_sub_lst.extend(filtered_heads)

    return rank_sub_lst.index(true_head)


def count_occurence(s, r, o, t):
    o_scores = []
    s_scores = []
    sub_rel_hist = sub_rel_to_ob[(s, r)]
    sub_obj_hist = sub_to_ob[s]
    rel_obj_hist = rel_to_ob[r]

    obj_rel_hist = obj_rel_to_sub[(o, r)]
    obj_sub_hist = ob_to_sub[o]
    rel_sub_hist = rel_to_sub[r]


    one=two=three=False
    if o in get_history_within_distance(sub_rel_hist, args.train_seq_len, t, args.future):
        o_scores.append(1)
        one = True
    if o in get_history_within_distance(sub_obj_hist, args.train_seq_len, t, args.future):
        o_scores.append(2)
        if not one:
            o_scores.append(5)
        two = True
    if o in get_history_within_distance(rel_obj_hist, args.train_seq_len, t, args.future):
        o_scores.append(3)
        if not one:
            o_scores.append(6)
        three = True
    if not one and not two and not three:
        o_scores.append(4)

    one=two=three=False
    if s in get_history_within_distance(obj_rel_hist, args.train_seq_len, t, args.future):
        s_scores.append(1)
        one = True
    if s in get_history_within_distance(obj_sub_hist, args.train_seq_len, t, args.future):
        s_scores.append(2)
        if not one:
            s_scores.append(5)
        two = True
    if s in get_history_within_distance(rel_sub_hist, args.train_seq_len, t, args.future):
        s_scores.append(3)
        if not one:
            s_scores.append(6)
        three = True
    if not one and not two and not three:
        s_scores.append(4)

    return s_scores, o_scores


def rank(s, r, o, t):
    s_string = id2ent[s]
    r_string = id2rel[r]
    o_string = id2ent[o]

    if debug: print("Triple: {}\t{}\t{}".format(s_string, r_string, o_string))
    rank_obj = rank_tail(s, r, t, o)
    rank_sub = rank_head(o, r, t, s)
    # pdb.set_trace()
    # pred_list.append([s, r, o, t, 's', rank_sub])
    # pred_list.append([s, r, o, t, 'o', rank_obj])
    return [rank_obj + 1, rank_sub + 1]


def calc_percentage_target_in_different_group():
    pass


def calc_metrics(rankings):
    rankings = np.array(rankings)
    mrr = np.mean(1 / np.array(rankings))
    hit_1 = np.mean((rankings <= 1))
    hit_3 = np.mean((rankings <= 3))
    hit_10 = np.mean((rankings <= 10))
    print("MRR: {}".format(mrr))
    print("HIT@10: {}".format(hit_10))
    print("HIT@3: {}".format(hit_3))
    print("HIT@1: {}".format(hit_1))


def count_repetitions():
    train_data_list = [tuple(i) for i in train_data.tolist()]
    val_data_list = [tuple(i) for i in val_data.tolist()]
    test_data_list = [tuple(i) for i in test_data.tolist()]
    train_data_set = set(train_data_list)
    val_data_set = set(val_data_list)
    test_data_set = set(test_data_list)
    print("Repetition inside train set:")
    print(len(train_data_list) - len(train_data_set))
    print("Repetition inside val set:")
    print(len(val_data_list) - len(val_data_set))
    print("Repetition inside test set:")
    print(len(test_data_list) - len(test_data_set))

    print("Repetition between train set and val set:")
    print(len(train_data_set.intersection(val_data_set)))
    print("Repetition between train set and test set:")
    print(len(train_data_set.intersection(test_data_set)))
    print("Repetition between val set and test set:")
    print(len(val_data_set.intersection(test_data_set)))


def get_results():
    rankings = []
    i = 0
    print(len(test_data))
    for quad in test_data:
        s, r, o, t = tuple(quad)
        rankings.extend(rank(s, r, o, t))

        if i % 100000 == 0:
            if debug: pdb.set_trace()
            calc_metrics(rankings)
            print()
            # print(np.mean((np.array(rankings) <= 10)))
            pass
        i += 1
        # if i == 3: break
    dataset = args.dataset.split("/")[-1]
    # with open("{}_greedy.pt".format(dataset), 'wb') as filehandle:
    #     pickle.dump(pred_list, filehandle)
    calc_metrics(rankings)


def count_rank_vs_category():
    s_scores = []
    o_scores = []
    i = 0
    for quad in test_data:
        s, r, o, t = tuple(quad)
        s_score, o_score = count_occurence(s, r, o, t)

        s_scores.extend(s_score)
        o_scores.extend(o_score)
        i += 1

    o_scores = np.array(o_scores)
    score_1 = np.sum(o_scores == 1)
    score_2 = np.sum(o_scores == 2)
    score_3 = np.sum(o_scores == 3)
    score_4 = np.sum(o_scores == 4)
    score_5 = np.sum(o_scores == 5)
    score_6 = np.sum(o_scores == 6)
    print("Object scores:")
    print("Score: 1, ratio: {}".format(score_1 / len(test_data)))
    print("Score: 2, ratio: {}".format(score_2 / len(test_data)))
    print("Score: 3, ratio: {}".format(score_3 / len(test_data)))
    print("Score: 4, ratio: {}".format(score_4 / len(test_data)))
    print("Score: 5, ratio: {}".format(score_5 / len(test_data)))
    print("Score: 6, ratio: {}".format(score_6 / len(test_data)))

    s_scores = np.array(s_scores)
    score_1 = np.sum(s_scores == 1)
    score_2 = np.sum(s_scores == 2)
    score_3 = np.sum(s_scores == 3)
    score_4 = np.sum(s_scores == 4)
    score_5 = np.sum(s_scores == 5)
    score_6 = np.sum(s_scores == 6)
    print("Subject scores:")
    print("Score: 1, ratio: {}".format(score_1 / len(test_data)))
    print("Score: 2, ratio: {}".format(score_2 / len(test_data)))
    print("Score: 3, ratio: {}".format(score_3 / len(test_data)))
    print("Score: 4, ratio: {}".format(score_4 / len(test_data)))
    print("Score: 5, ratio: {}".format(score_5 / len(test_data)))
    print("Score: 6, ratio: {}".format(score_6 / len(test_data)))

def get_true_head_and_tail_all():
    true_heads = defaultdict(lambda: defaultdict(list))
    true_tails = defaultdict(lambda: defaultdict(list))
    times = list(graph_dict_train.keys())
    for t in times:
        for g in graph_dict_train[t], graph_dict_val[t], graph_dict_test[t]:
            # triples.append(torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1))
            subjects = [g.ids[x] for x in g.edges()[0].tolist()]
            objects = [g.ids[x] for x in g.edges()[1].tolist()]
            relations = g.edata['type_s'].tolist()
            for s, r, o in zip(subjects, relations, objects):
                true_heads[t][(o, r)].append(s)
                true_tails[t][(s, r)].append(o)
        for o, r in true_heads[t].keys():
            true_heads[t][(o, r)] = list(set(true_heads[t][(o, r)]))
        for s, r in true_tails[t].keys():
            true_tails[t][(s, r)] = list(set(true_tails[t][(s, r)]))

    return true_heads, true_tails


if __name__ == '__main__':
    args = process_args()
    filtered = args.filtered
    torch.manual_seed(args.seed)

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))
    debug = args.debug
    lam = args.inv_temperature
    vote = args.vote
    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

    id2ent, id2rel = id2entrel(args.dataset, num_rels)
    entities = list(range(num_ents))
    if args.dataset_dir == 'interpolation':
        graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)
    true_heads, true_tails = get_true_head_and_tail_all()
    train_data, train_times = load_quadruples(args.dataset, 'train.txt')
    val_data, val_times = load_quadruples(args.dataset, 'valid.txt')
    # test_data, test_times = load_quadruples(args.dataset, 'train.txt')
    test_data, test_times = load_quadruples(args.dataset, 'test.txt')
    if "gdelt" in args.dataset:
        train_set = set([tuple(x) for x in train_data])
        train_data = list(train_set)
        test_set = set([tuple(x) for x in test_data])
        train_test_intersect = train_set.intersection(test_set)
        test_data = [i for i in test_set if i not in train_test_intersect]

    # test_data, test_times = load_quadruples(args.dataset, 'test.txt')
    count_repetitions()
    exit()
    # total_data = np.concatenate(train_data, val_data, test_data, axis=None)

    # time2triples = load_quadruples_interpolation(args.dataset, 'train.txt', 'valid.txt', 'test.txt', total_times)

    triple_freq_per_time_step, ent_pair_freq_per_time_step, sub_freq_per_time_step, obj_freq_per_time_step, \
        rel_freq_per_time_step, sub_rel_freq_per_time_step, obj_rel_freq_per_time_step = count_entity_freq_per_train_graph(train_data)
    sub_rel_to_ob, obj_rel_to_sub, sub_to_ob, ob_to_sub, rel_to_ob, rel_to_sub = construct_ref_data(train_data)
    # count_rank_vs_category()
    get_results()
