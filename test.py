# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
import pdb
from models.TKG_VRE import TKG_VAE
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_lightning.logging import TestTubeLogger
import pandas as pd
from utils.utils import MyTestTubeLogger
import json

if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))

    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()
    # args.n_gpu = torch.cuda.device_count()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    if args.dataset == 'extrapolation/ICEWS14':
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'test.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'tain.txt', 'test.txt')
    else:
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'valid.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'valid.txt','test.txt')

    graph_dict_train, graph_dict_dev, graph_dict_test = build_extrapolation_time_stamp_graph(args)
    pdb.set_trace()
    model = TKG_VAE(args, num_ents, num_rels, graph_dict_train, graph_dict_test, train_times, valid_times)
    state_dict = torch.load(args.checkpoint_path)
    TKG_VAE.load_state_dict(state_dict['state_dict'])
    # model = TKG_VAE_Module.load_from_checkpoint(args.checkpoint_path, num_ents, num_rels, graph_dict_train, graph_dict_dev, graph_dict_test, train_times, valid_times, test_times)
    trainer = Trainer(gpus=args.n_gpu)
    trainer.test(model)
