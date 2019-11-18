import numpy as np
import torch
import pdb

def make_batch(a, n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]


def samples_labels(g_batched_list, negative_rate, use_gpu):
    samples = []
    labels = []
    for g in g_batched_list:
        triples = torch.stack([g.edges()[0], g.edata['type_s'].cpu(), g.edges()[1]]).transpose(0, 1)
        sample, label = negative_sampling(triples, len(g.nodes()), negative_rate)
        sample = torch.from_numpy(sample)
        label = torch.from_numpy(label)
        if use_gpu:
            sample = sample.cuda()
            label = label.cuda()
            move_dgl_to_cuda(g)
        samples.append(sample)
        labels.append(label)
    return samples, labels


def move_dgl_to_cuda(g):
    g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
    g.edata.update({k: cuda(g.edata[k]) for k in g.edata})


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

# TODO: refine sampling procedure
def negative_sampling(pos_samples, num_entities, negative_rate):
    size_of_batch = pos_samples.shape[0]
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entities, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    return np.concatenate((pos_samples, neg_samples)), labels


def _nll_bernoulli(theta, x):
    return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))


def _kld_gauss(mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD"""
    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element)