from torch import nn
import torch.nn.functional as F
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import samples_labels, reparametrize
from utils.scores import *
from utils.evaluation import calc_metrics
from argparse import Namespace
from models.TKG_Module import TKG_Module

class HYTE(TKG_Module):
    def __init__(self, args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times):
        super(HYTE, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_test, train_times,
                                   valid_times)
