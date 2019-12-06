import numpy as np
import torch
import pdb
from pytorch_lightning.logging import TestTubeLogger
import os
import json



def move_dgl_to_cuda(g):
    g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
    g.edata.update({k: cuda(g.edata[k]) for k in g.edata})

def filter_none(l):
    return list(filter(lambda x: x is not None, l))

def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

'''
# TODO: check if the sampled triples are in the KB
def negative_sampling(triples, num_entities, negative_rate, num_pos_facts, use_gpu):
    size_of_batch = min(triples.shape[0], num_pos_facts)
    if num_pos_facts < triples.shape[0]:
        rand_idx = torch.randperm(triples.shape[0])
        pos_samples = triples[rand_idx[:num_pos_facts]]
    else:
        pos_samples = triples
    # pos_samples = pos_samples
    # size_of_batch = pos_samples.shape[0]

    num_to_generate = size_of_batch * negative_rate
    neg_samples = pos_samples.repeat(negative_rate, 1)
    labels = torch.zeros(size_of_batch * (negative_rate + 1))
    labels[: size_of_batch] = 1
    values = torch.randint(low=0, high=num_entities, size=(num_to_generate,))
    if use_gpu:
        labels = labels.cuda()
        values = values.cuda()
    choices = torch.rand(num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    return torch.cat((pos_samples, neg_samples), dim=0), labels
'''

class MyTestTubeLogger(TestTubeLogger):
    def __init__(self, *args, **kwargs):
        super(MyTestTubeLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, args):
        config_path = self.experiment.get_data_path(self.experiment.name, self.experiment.version)
        with open(os.path.join(config_path, 'config.json'), 'w') as configfile:
            configfile.write(json.dumps(vars(args), indent=2, sort_keys=True))


def reparametrize(mean, std):
    """using std to sample"""
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm
