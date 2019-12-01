import numpy as np
import torch
import pdb
from utils.utils import cuda
import time

class CorruptTriples():
    def __init__(self, args, graph_dict_total):
        self.args = args
        self.negative_rate = args.negative_rate
        self.use_cuda = args.use_cuda
        self.num_pos_facts = args.num_pos_facts
        self.graph_dict_total = graph_dict_total
        self.get_true_hear_and_tail()

    def get_true_hear_and_tail(self):
        self.true_heads = dict()
        self.true_tails = dict()
        start = time.time()
        for t, g in self.graph_dict_total.items():
            triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            true_head, true_tail = self.get_true_head_and_tail_per_graph(triples)
            self.true_heads[t] = true_head
            self.true_tails[t] = true_tail
        print("Collected true head and true tails: {}".format(time.time() - start))

    def samples_labels(self, t_list, g_batched_list, val=False):
        samples = []
        labels = []
        for t, g in zip(t_list, g_batched_list):
            sample, label = self.single_graph_sampling(t, g, val)
            samples.append(sample)
            labels.append(label)
        return samples, labels

    def single_graph_sampling(self, t, g, val=False):
        t = t.item()
        triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)

        if not val:
            sample, label = self.negative_sampling(self.true_heads[t], self.true_tails[t], triples, len(g.nodes()))
            sample, label = torch.from_numpy(sample), torch.from_numpy(label)
        else:
            sample, label = triples, torch.ones(triples.shape[0])

        if self.use_cuda:
            sample = cuda(sample)
            label = cuda(label)
        return sample, label

    def corrupt_triple(self, h, r, t, true_head, true_tail, num_entities, tail=True):
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_rate:
            negative_sample = np.random.randint(num_entities, size=self.negative_rate)
            if tail:
                mask = np.in1d(
                    negative_sample,
                    true_tail[(h, r)],
                    assume_unique=True,
                    invert=True
                )
            else:
                mask = np.in1d(
                    negative_sample,
                    true_head[(r, t)],
                    assume_unique=True,
                    invert=True
                )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        return np.concatenate(negative_sample_list)[:self.negative_rate]

    def negative_sampling(self, true_head, true_tail, triples, num_entities):
        size_of_batch = min(triples.shape[0], self.num_pos_facts)
        if self.num_pos_facts < triples.shape[0]:
            rand_idx = torch.randperm(triples.shape[0])
            triples = triples[rand_idx[:self.num_pos_facts]]

        neg_samples = np.tile(triples, (1, self.negative_rate)).reshape(-1, 3)
        labels = np.zeros(size_of_batch * (self.negative_rate + 1), dtype=np.float32)
        labels[: size_of_batch] = 1
        choices = np.random.uniform(size=size_of_batch)
        tails = choices <= 0.5

        for i in range(size_of_batch):
            h, r, t = triples[i]
            h, r, t = h.item(), r.item(), t.item()
            negative_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, tails[i])
            if tails[i]:
                neg_samples[i * self.negative_rate : (i + 1) * self.negative_rate, 2] = negative_samples
            else:
                neg_samples[i * self.negative_rate : (i + 1) * self.negative_rate, 0] = negative_samples


        shuffled_idx = np.random.permutation(size_of_batch)
        return np.concatenate((triples, neg_samples))[shuffled_idx], labels[shuffled_idx]

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

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail
