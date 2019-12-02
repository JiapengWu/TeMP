import numpy as np
import torch
import pdb
from utils.utils import cuda
import time
import torch.nn.functional as F


class CorruptTriples:
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

    def samples_labels_train(self, t_list, g_batched_list):
        samples = []
        neg_tail_samples = []
        neg_head_samples = []
        labels = []
        for t, g in zip(t_list, g_batched_list):
            sample, neg_tail_sample, neg_head_sample, label = self.single_graph_negative_sampling(t, g)
            samples.append(sample)
            neg_tail_samples.append(neg_tail_sample)
            neg_head_samples.append(neg_head_sample)
            labels.append(label)
        return samples, neg_tail_samples, neg_head_samples, labels

    def single_graph_negative_sampling(self, t, g):
        t = t.item()
        triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
        sample, neg_tail_sample, neg_head_sample, label = self.negative_sampling(self.true_heads[t], self.true_tails[t], triples, len(g.nodes()))
        neg_tail_sample, neg_head_sample, label = torch.from_numpy(neg_tail_sample), torch.from_numpy(neg_head_sample), torch.from_numpy(label)
        if self.use_cuda:
            sample, neg_tail_sample, neg_head_sample, label = cuda(sample), cuda(neg_tail_sample), cuda(neg_head_sample), cuda(label)
        return sample, neg_tail_sample, neg_head_sample, label

    def negative_sampling(self, true_head, true_tail, triples, num_entities):
        size_of_batch = min(triples.shape[0], self.num_pos_facts)
        if self.num_pos_facts < triples.shape[0]:
            rand_idx = torch.randperm(triples.shape[0])
            triples = triples[rand_idx[:self.num_pos_facts]]
        # pdb.set_trace()
        neg_tail_samples = np.zeros((size_of_batch, 1 + self.negative_rate), dtype=int)
        neg_head_samples = np.zeros((size_of_batch, 1 + self.negative_rate), dtype=int)
        neg_tail_samples[:, 0] = triples[:, 2]
        neg_head_samples[:, 0] = triples[:, 0]

        # labels = np.zeros((size_of_batch, 1 + self.negative_rate), dtype=int)
        # labels[:, 0] = 1
        labels = np.zeros(size_of_batch, dtype=int)

        for i in range(size_of_batch):
            h, r, t = triples[i]
            h, r, t = h.item(), r.item(), t.item()
            tail_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, True)
            head_samples = self.corrupt_triple(h, r, t, true_head, true_tail, num_entities, False)

            neg_tail_samples[i, 1:] = tail_samples
            neg_head_samples[i, 1:] = head_samples

        return triples, neg_tail_samples, neg_head_samples, labels

    def sample_labels_val(self, g_batched_list):
        samples = []
        labels = []
        for g in g_batched_list:
            sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(sample.shape[0])
            if self.use_cuda:
                sample = cuda(sample)
                label = cuda(label)
            samples.append(sample)
            labels.append(label)
        return samples, labels

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
