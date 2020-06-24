import numpy as np
import torch
import pdb
from utils.utils import cuda


class CorruptTriples:
    def __init__(self, args, graph_dict_train):
        self.args = args
        self.negative_rate = args.negative_rate
        self.use_cuda = args.use_cuda
        self.num_pos_facts = args.num_pos_facts
        self.graph_dict_train = graph_dict_train
        self.get_true_hear_and_tail()

    def get_true_hear_and_tail(self):
        self.true_heads_train = dict()
        self.true_tails_train = dict()
        for t, g in self.graph_dict_train.items():
            triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            true_head, true_tail = self.get_true_head_and_tail_per_graph(triples)
            self.true_heads_train[t] = true_head
            self.true_tails_train[t] = true_tail

    # TODO: fix negative sampling to include all the nodes
    def single_graph_negative_sampling(self, t, g, num_ents):
        t = t.item()
        triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
        sample, neg_tail_sample, neg_head_sample, label = self.negative_sampling(self.true_heads_train[t], self.true_tails_train[t], triples, num_ents, g)

        neg_tail_sample, neg_head_sample, label = torch.from_numpy(neg_tail_sample), torch.from_numpy(neg_head_sample), torch.from_numpy(label)
        if self.use_cuda:
            sample, neg_tail_sample, neg_head_sample, label = cuda(sample), cuda(neg_tail_sample), cuda(neg_head_sample), cuda(label)
        return sample, neg_tail_sample, neg_head_sample, label

    def negative_sampling(self, true_head, true_tail, triples, num_entities, g):
        size_of_batch = min(triples.shape[0], self.num_pos_facts)
        if self.num_pos_facts < triples.shape[0]:
            rand_idx = torch.randperm(triples.shape[0])
            triples = triples[rand_idx[:self.num_pos_facts]]
        neg_tail_samples = np.zeros((size_of_batch, 1 + self.negative_rate), dtype=int)
        neg_head_samples = np.zeros((size_of_batch, 1 + self.negative_rate), dtype=int)
        neg_tail_samples[:, 0] = triples[:, 2]
        neg_head_samples[:, 0] = triples[:, 0]

        labels = np.zeros(size_of_batch, dtype=int)

        for i in range(size_of_batch):
            h, r, t = triples[i]
            h, r, t = h.item(), r.item(), t.item()
            tail_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, g, True)
            head_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, g, False)

            neg_tail_samples[i][0] = g.ids[triples[i][2].item()]
            neg_head_samples[i][0] = g.ids[triples[i][0].item()]
            neg_tail_samples[i, 1:] = tail_samples
            neg_head_samples[i, 1:] = head_samples

        return triples, neg_tail_samples, neg_head_samples, labels

    def corrupt_triple(self, h, r, t, true_head, true_tail, num_entities, g, tail=True):
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_rate:
            negative_sample = np.random.randint(num_entities, size=self.negative_rate)

            if tail:
                mask = np.in1d(
                    negative_sample,
                    [g.ids[i.item()] for i in true_tail[(h, r)]],
                    assume_unique=True,
                    invert=True
                )
            else:
                mask = np.in1d(
                    negative_sample,
                    [g.ids[i.item()] for i in true_head[(r, t)]],
                    assume_unique=True,
                    invert=True
                )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:self.negative_rate]

    @staticmethod
    def get_true_head_and_tail_per_graph(triples):
        true_head = {}
        true_tail = {}
        for head, relation, tail in triples:
            head, relation, tail = head.item(), relation.item(), tail.item()
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        # this is correct
        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail
