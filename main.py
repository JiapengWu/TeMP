# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
from models.TKG_VRE import TKG_VAE
from baselines.Static import Static
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
from ablation.RecurrentRGCN import RecurrentRGCN
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_lightning.logging import TestTubeLogger
from utils.utils import MyTestTubeLogger
import json
from pytorch_lightning.callbacks import ModelCheckpoint
import pdb

if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))

    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()
    # args.n_gpu = torch.cuda.device_count()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

    if args.dataset_dir == 'extrapolation':
        graph_dict_train, graph_dict_val, graph_dict_test = build_extrapolation_time_stamp_graph(args)
    elif args.dataset_dir == 'interpolation':
        graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    # pdb.set_trace()
    # graph_dict_total = {**graph_dict_train, **graph_dict_dev, **graph_dict_test}

    module = {
              'VKGRNN': TKG_VAE,
              "Static": Static,
              "DE": DiachronicEmbedding,
              "SRGCN": StaticRGCN,
              "RRGCN": RecurrentRGCN
              }[args.module]

    model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='min'
    )

    tt_logger = MyTestTubeLogger(
        save_dir="experiments",
        name="{}-{}-{}".format(args.module, args.dataset.split('/')[-1], args.score_function),
        debug=False,
        version=time.strftime('%Y%m%d%H%M'),
        create_git_tag=True
    )

    checkpoint_path = os.path.join(tt_logger.experiment.get_data_path(tt_logger.experiment.name, tt_logger.experiment.version), "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=''
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
                      nb_sanity_val_steps=1 if args.debug else 5,
                      early_stop_callback=early_stop_callback,
                      train_percent_check=0.1 if args.debug else 1.0,
                      checkpoint_callback=checkpoint_callback
                      # print_nan_grads=True
                      # truncated_bptt_steps=4
                      )

    trainer.fit(model)
    trainer.test(model)


