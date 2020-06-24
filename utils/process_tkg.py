from random import *
from collections import defaultdict as ddict
import pdb
import os
from process_interpolation_dataset import get_args

def create_year2id(triple_time):
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

    return year2id, year_list


def create_id_labels(triple_time, year2id):
    YEARMAX = 3000
    YEARMIN = -50

    inp_idx, start_idx, end_idx = [], [], []

    for k, v in triple_time.items():
        # pdb.set_trace()
        start = v[0].split('-')[0]
        end = v[1].split('-')[0]
        if start == '####':
            start = YEARMIN
        elif start.find('#') != -1 or len(start) != 4:
            continue

        if end == '####':
            end = YEARMAX
        elif end.find('#') != -1 or len(end) != 4:
            continue

        start = int(start)
        end = int(end)

        if start > end:
            end = YEARMAX
        inp_idx.append(k)
        if start == YEARMIN:
            start_idx.append(0)
        else:
            for key, lbl in sorted(year2id.items(), key=lambda x: x[1]):
                if start >= key[0] and start <= key[1]:
                    start_idx.append(lbl)

        if end == YEARMAX:
            end_idx.append(len(year2id.keys()) - 1)
        else:
            for key, lbl in sorted(year2id.items(), key=lambda x: x[1]):
                if end >= key[0] and end <= key[1]:
                    end_idx.append(lbl)
        # pdb.set_trace()
    return inp_idx, start_idx, end_idx

def write_processed_files(triples, triple_time, year2id, mode):
    inp_idx, start_idx, end_idx = create_id_labels(triple_time, year2id)
    print(mode)
    with open(os.path.join(output_dir, "{}.txt".format(mode)), "w") as f:
        for triple_id, start_id, end_id in zip(inp_idx, start_idx, end_idx):
            triple = triples[triple_id]
            if start_id < end_id:
                for t in range(start_id, end_id + 1):
                    # pdb.set_trace()
                    f.write("{}\t{}\t{}\t{}\n".format(triple[0], triple[1], triple[2], t))

def load_data():
    # triple_set = []
    # with open(os.path.join(input_dir, "triple2id.txt"), 'r') as filein:
    #     for line in filein:
    #         tup = (int(line.split()[0].strip()), int(line.split()[1].strip()), int(line.split()[2].strip()))
    #         triple_set.append(tup)
    # triple_set = set(triple_set)

    train_triples = []
    val_triples = []
    test_triples = []
    train_triple_time, entity_time = dict(), dict()
    val_triple_time, test_triple_time = dict(), dict()
    inp_idx, start_idx, end_idx, labels = ddict(list), ddict(list), ddict(list), ddict(list)
    max_ent, max_rel = 0, 0

    for mode, triples, triple_time in zip(["train.txt", "valid.txt", "test.txt"], [train_triples, val_triples, test_triples], [train_triple_time, val_triple_time, test_triple_time]):
        count = 0
        with open(os.path.join(input_dir, mode), 'r') as filein:
            for line in filein:
                triples.append([int(x.strip()) for x in line.split()[0:3]])
                triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
                count += 1

    ent_idx = []
    with open(os.path.join(input_dir, "entity2id.txt"), 'r', encoding="utf-8") as filein2:
        for line in filein2:
            ent_idx.append(int(line.split('\t')[1]))
            max_ent = max_ent + 1
    with open(os.path.join(input_dir, 'relation2id.txt'), 'r') as filein3:
        for line in filein3:
            max_rel = max_rel + 1

    year2id, year_list = create_year2id(train_triple_time)

    for mode, triples, triple_time in zip(["train", "valid", "test"], [train_triples, val_triples, test_triples], [train_triple_time, val_triple_time, test_triple_time]):
        write_processed_files(triples, triple_time, year2id, mode)

    with open(os.path.join(output_dir, 'stat.txt'), "w") as f:
        f.write('{}\t{}\t{}'.format(max_ent, max_rel, len(year2id)))


if __name__ == '__main__':
    args = get_args()
    input_dir = os.path.join('raw', args.dataset_dir)
    output_dir = os.path.join('interpolation', args.dataset_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    load_data()