# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
from previous.TKG_VRE import TKG_VAE
from baselines.Static import Static
from baselines.Simple import SimplE
from baselines.Hyte import Hyte
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
from models.DynamicRGCN import DynamicRGCN
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from utils.utils import MyTestTubeLogger
import json
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)

    if args.config:
        args_json = json.load(open(args.config))
        args.__dict__.update(dict(args_json))

    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

    if args.dataset_dir == 'extrapolation':
        graph_dict_train, graph_dict_val, graph_dict_test = build_extrapolation_time_stamp_graph(args)
    elif args.dataset_dir == 'interpolation':
        graph_dict_train, graph_dict_val, graph_dict_test = build_interpolation_graphs(args)

    module = {
              'VKGRNN': TKG_VAE,
              "Simple": SimplE,
              "Static": Static,
              "DE": DiachronicEmbedding,
              "Hyte": Hyte,
              "SRGCN": StaticRGCN,
              "RRGCN": DynamicRGCN,
              "DRGCN": DynamicRGCN
              }[args.module]

    model = module(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    early_stop_callback = EarlyStopping(
        monitor='mrr',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='max'
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
        monitor='mrr',
        mode='max',
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
                      distributed_backend=args.distributed_backend,
                      nb_sanity_val_steps=1 if args.debug else 1,
                      early_stop_callback=early_stop_callback,
                      train_percent_check=0.1 if args.debug else 1.0,
                      checkpoint_callback=checkpoint_callback
                      )

    trainer.fit(model)
    trainer.test(model)


