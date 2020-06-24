import argparse
import os
import re

def get_args():
    parser = argparse.ArgumentParser(description='TKG-VAE')
    parser.add_argument("--dataset-dir", '-d', type=str, default='wiki')
    return parser.parse_args()

def remove_redundant(train_data, other_data):
    return [i for i in other_data if i not in train_data]

def create_ent_rel_to_idx():
    times = []
    entities = []
    relations = []
    train_triples = []
    valid_triples = []
    test_triples = []
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
                time = line_split[3]
                time = int(re.sub(r'-', '', time))
                times.append(time)
                triple_lst.append((head, rel, tail, time))

    # if "gdelt" in args.dataset_dir:
    #     print("cleaning...")
    #     train_set = set(train_triples)
    #     val_set = set(valid_triples)
    #     test_set = set(test_triples)
    #     train_triples = list(train_set)
    #     valid_triples = list(val_set)
    #     test_triples = list(test_set)
    #     train_val_intersect = train_set.intersection(val_set)
    #     train_test_intersect = train_set.intersection(test_set)
    #     print("val")
    #     valid_triples = [i for i in valid_triples if i not in train_val_intersect]
    #     print("test")
    #     test_triples = [i for i in test_triples if i not in train_test_intersect]

    return list(set(times)), list(set(entities)), list(set(relations)), train_triples, valid_triples, test_triples


def write_stats_idx():
    with open(os.path.join(output_dir, 'entity2id.txt'), "w") as f:
        for i, ent in enumerate(entities):
            f.write("{}\t{}\n".format(ent, i))

    with open(os.path.join(output_dir, 'relation2id.txt'), "w") as f:
        for i, rel in enumerate(relations):
            f.write("{}\t{}\n".format(rel, i))

    with open(os.path.join(output_dir, 'stat.txt'), "w") as f:
        f.write('{}\t{}\t{}'.format(len(entities), len(relations), len(times)))


def write_processed_files():
    for data_split, triple_lst in zip(['train', 'valid', 'test'], [train_triples, valid_triples, test_triples]):
        with open(os.path.join(output_dir, "{}.txt".format(data_split)), "w") as f:
            for head, rel, tail, time in triple_lst:
                f.write("{}\t{}\t{}\t{}\n".format(ent2id[head], rel2id[rel], ent2id[tail], time2id[time]))


if __name__ == '__main__':
    args = get_args()
    input_dir = os.path.join('raw', args.dataset_dir)
    output_dir = os.path.join('interpolation', args.dataset_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    times, entities, relations, train_triples, valid_triples, test_triples = create_ent_rel_to_idx()
    times.sort()

    num_times = len(times)
    num_ents = len(entities)
    num_rels = len(relations)

    write_stats_idx()

    time2id = {k: v for v, k in enumerate(times)}
    ent2id = {k: v for v, k in enumerate(entities)}
    rel2id = {k: v for v, k in enumerate(relations)}

    write_processed_files()