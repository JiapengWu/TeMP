import argparse
import os
import pdb
import string
import re
from process_interpolation_dataset import *
from collections import Counter, defaultdict

def create_ent_rel_to_idx():
    entities = []
    relations = []
    train_triples = []
    valid_triples = []
    test_triples = []
    times = []
    with open(os.path.join(input_dir, "train.txt"), "r") as f:
        for line in f:
            line_split = line.strip().split('\t')
            try:
                start_time = int(line_split[3].split('-')[0])
                times.append(start_time)
            except:
                pass

            try:
                end_time = int(line_split[4].split('-')[0])
                times.append(end_time)
            except:
                pass
    times = list(set(times))
    min_time = min(times)
    max_time = max(times)

    years = list()
    for data_split, triple_lst in zip(['train', 'valid', 'test'], [train_triples, valid_triples, test_triples]):
        with open(os.path.join(input_dir, "{}.txt".format(data_split)), "r") as f:
            for line in f:
                line_split = line.strip().split('\t')
                head = line_split[0]
                tail = line_split[2]
                rel = line_split[1]
                entities.append(head)
                entities.append(tail)
                relations.append(rel)
                start_time = line_split[3].split('-')[0]
                end_time = line_split[4].split('-')[0]

                if start_time == '####':
                    start_time = min_time
                if end_time == '####':
                    end_time = max_time
                start_time = int(start_time)
                end_time = int(end_time)

                if start_time > end_time:
                    end_time = max_time

                years.extend(list(range(start_time, end_time + 1)))
                for t in range(start_time, end_time + 1):
                    triple_lst.append((head, rel, tail, t))
    year_freq = Counter(years)
    years2id = defaultdict(list)
    sum_freq = 0
    id = 0

    for year in sorted(list(year_freq.keys())):

        pdb.set_trace()
        sum_freq += year_freq[year]
        if sum_freq >= 300:
            sum_freq = 0
            id += 1
        else:
            years2id[id].append(year)

    return list(set(times)), list(set(entities)), list(set(relations)), train_triples, valid_triples, test_triples


if __name__ == '__main__':
    args = get_args()
    input_dir = os.path.join('raw', args.dataset_dir)
    output_dir = os.path.join('interpolation', args.dataset_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    times, entities, relations, train_triples, valid_triples, test_triples = create_ent_rel_to_idx()
    times.sort()
    # pdb.set_trace()

    num_times = len(times)
    num_ents = len(entities)
    num_rels = len(relations)

    write_stats_idx()

    time2id = {k: v for v, k in enumerate(times)}
    ent2id = {k: v for v, k in enumerate(entities)}
    rel2id = {k: v for v, k in enumerate(relations)}

    write_processed_files()