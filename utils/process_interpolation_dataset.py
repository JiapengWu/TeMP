import argparse
import os
import pdb
import string
import re

def get_args():
    parser = argparse.ArgumentParser(description='TKG-VAE')
    parser.add_argument("--dataset-dir", '-d', type=str, default='extrapolation')
    return parser.parse_args()


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
    # pdb.set_trace()

    num_times = len(times)
    num_ents = len(entities)
    num_rels = len(relations)

    write_stats_idx()

    time2id = {k: v for v, k in enumerate(times)}
    ent2id = {k: v for v, k in enumerate(entities)}
    rel2id = {k: v for v, k in enumerate(relations)}

    write_processed_files()