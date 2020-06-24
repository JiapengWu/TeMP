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


def create_year2id(self, triple_time):
    year2id = dict()
    freq = ddict(int)
    count = 0
    year_list = []

    for k, v in triple_time.items():
        try:
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
        except:
            pdb.set_trace()

        if start.find('#') == -1 and len(start) == 4:
            year_list.append(int(start))
        if end.find('#') == -1 and len(end) == 4:
            year_list.append(int(end))

    # for k,v in entity_time.items():
    # 	start = v[0].split('-')[0]
    # 	end = v[1].split('-')[0]

    # 	if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
    # 	if end.find('#') == -1 and len(end) ==4: year_list.append(int(end))
    # 	# if int(start) > int(end):
    # 	# 	pdb.set_trace()

    year_list.sort()
    for year in year_list:
        freq[year] = freq[year] + 1

    # pdb.set_trace()

    year_class = []
    count = 0
    for key in sorted(freq.keys()):
        count += freq[key]
        if count > 300:
            year_class.append(key)
            count = 0
    prev_year = 0
    i = 0

    # pdb.set_trace()

    for i, yr in enumerate(year_class):
        year2id[(prev_year, yr)] = i
        prev_year = yr + 1
    year2id[(prev_year, max(year_list))] = i + 1
    self.year_list = year_list

    # pdb.set_trace()

    # for k,v in entity_time.items():
    # 	if v[0] == '####-##-##' or v[1] == '####-##-##':
    # 		continue
    # 	if len(v[0].split('-')[0])!=4 or len(v[1].split('-')[0])!=4:
    # 		continue
    # 	start = v[0].split('-')[0]
    # 	end = v[1].split('-')[0]
    # for start in start_list:
    # 	if start not in start_year2id:
    # 		start_year2id[start] = count_start
    # 		count_start+=1

    # for end in end_list:
    # 	if end not in end_year2id:
    # 		end_year2id[end] = count_end
    # 		count_end+=1

    return year2id

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