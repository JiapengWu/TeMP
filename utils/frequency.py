from collections import defaultdict

def count_entity_freq_per_train_graph(train_data):
    # train_graph_dict, _, _ = build_interpolation_graphs(args)
    triple_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    ent_pair_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    sub_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    obj_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    sub_rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    obj_rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))

    for quad in train_data:
        sub, rel, obj, tim = tuple(quad)
        # for triple in triples:
        # pdb.set_trace()
        triple_freq_per_time_step[(sub, rel, obj)][tim] += 1
        ent_pair_freq_per_time_step[(sub, obj)][tim] += 1
        sub_freq_per_time_step[sub][tim] += 1
        obj_freq_per_time_step[obj][tim] += 1
        rel_freq_per_time_step[rel][tim] += 1
        sub_rel_freq_per_time_step[(sub, rel)][tim] += 1
        obj_rel_freq_per_time_step[(obj, rel)][tim] += 1

    return triple_freq_per_time_step, ent_pair_freq_per_time_step, sub_freq_per_time_step, obj_freq_per_time_step, rel_freq_per_time_step, sub_rel_freq_per_time_step, obj_rel_freq_per_time_step

def temp_func():
    return defaultdict(int)

def count_freq_per_time(train_data):
    # train_graph_dict, _, _ = build_interpolation_graphs(args)
    triple_freq_per_time_step = defaultdict(temp_func)
    ent_pair_freq_per_time_step = defaultdict(temp_func)
    sub_freq_per_time_step = defaultdict(temp_func)
    obj_freq_per_time_step = defaultdict(temp_func)
    rel_freq_per_time_step = defaultdict(temp_func)
    sub_rel_freq_per_time_step = defaultdict(temp_func)
    obj_rel_freq_per_time_step = defaultdict(temp_func)

    for quad in train_data:
        sub, rel, obj, tim = tuple(quad)
        triple_freq_per_time_step[tim][(sub, rel, obj)] += 1
        ent_pair_freq_per_time_step[tim][(sub, obj)] += 1
        sub_freq_per_time_step[tim][sub] += 1
        obj_freq_per_time_step[tim][obj] += 1
        rel_freq_per_time_step[tim][rel] += 1
        sub_rel_freq_per_time_step[tim][(sub, rel)] += 1
        obj_rel_freq_per_time_step[tim][(obj, rel)] += 1

    return triple_freq_per_time_step, ent_pair_freq_per_time_step, sub_freq_per_time_step, obj_freq_per_time_step, rel_freq_per_time_step, sub_rel_freq_per_time_step, obj_rel_freq_per_time_step

def calc_aggregated_statistics(stats_per_time_agg, items, stats_per_time, target_time, cur_time):
    for item in items:
        if item in stats_per_time[cur_time].keys():
            stats_per_time_agg[target_time][item] += stats_per_time[cur_time][item]

def count_aggregated_freq_per_time(train_data, train_len, future=False):

    triple_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    ent_pair_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    sub_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    obj_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    sub_rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))
    obj_rel_freq_per_time_step = defaultdict(lambda: defaultdict(int))

    for quad in train_data:
        sub, rel, obj, tim = tuple(quad)
        upper = max(0, tim - train_seq_len + 1)


def get_history_within_distance(history, distance, cur_time, future):
    res = []
    for time in history.keys():
        time_diff = cur_time - time
        condition = time_diff <= distance and time_diff >= 0 if not future else abs(cur_time - time) <= distance
        if condition:
            # pdb.set_trace()
            res.extend(history[time])
    return set(res)


def construct_ref_data(train_data):
    sub_rel_to_ob = defaultdict(lambda: defaultdict(list))
    obj_rel_to_sub = defaultdict(lambda: defaultdict(list))
    sub_to_ob = defaultdict(lambda: defaultdict(list))
    ob_to_sub = defaultdict(lambda: defaultdict(list))
    rel_to_ob = defaultdict(lambda: defaultdict(list))
    rel_to_sub = defaultdict(lambda: defaultdict(list))
    for quad in train_data:
        s, r, o, t = tuple(quad)
        sub_rel_to_ob[(s, r)][t].append(o)
        obj_rel_to_sub[(o, r)][t].append(s)
        sub_to_ob[s][t].append(o)
        ob_to_sub[o][t].append(s)
        rel_to_sub[r][t].append(s)
        rel_to_ob[r][t].append(o)
    return sub_rel_to_ob, obj_rel_to_sub, sub_to_ob, ob_to_sub, rel_to_ob, rel_to_sub


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
