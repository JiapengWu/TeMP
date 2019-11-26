# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
import pdb
from models.TKG_VRE import TKG_VAE_Module
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
    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))

    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')
    if args.dataset == 'ICEWS14':
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'test.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'tain.txt', 'test.txt')
    else:
        train_data, train_times = load_quadruples(args.dataset, 'train.txt')
        valid_data, valid_times = load_quadruples(args.dataset, 'valid.txt')
        test_data, test_times = load_quadruples(args.dataset, 'test.txt')
        total_data, total_times = load_quadruples(args.dataset, 'train.txt', 'valid.txt','test.txt')

    print(train_times)
    print(valid_times)
    print(test_times)
    graph_dict_train, graph_dict_dev, graph_dict_test = build_time_stamp_graph(args)

    model = TKG_VAE_Module(args, num_ents, num_rels, graph_dict_train, graph_dict_dev, graph_dict_test, train_times, valid_times, test_times)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='min'
    )

    tt_logger = MyTestTubeLogger(
        save_dir="experiments",
        name="VKGRNN-{}-{}".format(args.dataset.split('/')[-1], args.score_function),
        debug=False,
        version=time.strftime('%Y%m%d%H%M'),
        create_git_tag=True
    )

    tt_logger.log_hyperparams(args)
    tt_logger.save()
    # most basic trainer, uses good defaults
    trainer = Trainer(logger=tt_logger, gpus=args.n_gpu,
                      gradient_clip_val=args.gradient_clip_val,
                      use_amp=args.use_amp,
                      amp_level=args.amp_level,
                      max_nb_epochs=args.max_nb_epochs,
                      # fast_dev_run=args.debug,
                      # log_gpu_memory='min_max' if args.debug else None,
                      distributed_backend=args.distributed_backend,
                      nb_sanity_val_steps=1,
                      early_stop_callback=early_stop_callback,
                      train_percent_check=0.1 if args.debug else 0
                      # print_nan_grads=args.debug
                      # truncated_bptt_steps=4
                      )
    trainer.fit(model)
    # trainer.test(model)


