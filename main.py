# from comet_ml import Experiment, ExistingExperiment
from utils.dataset import *
from utils.args import process_args
from baselines.Static import Static
from baselines.Simple import SimplE
from baselines.Hyte import Hyte
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
from models.DynamicRGCN import DynamicRGCN
from previous.TimeRGCN import TimeRGCN
from models.TimeDynamicRGCN import TimeDynamicRGCN
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from utils.utils import MyTestTubeLogger
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from models.BiDynamicRGCN import BiDynamicRGCN
from models.SelfAttentionRGCN import SelfAttentionRGCN
from models.BiSelfAttentionRGCN import BiSelfAttentionRGCN
from models.aggregator import Aggregator
from models.PostDynamicRGCN import ImputeDynamicRGCN, PostDynamicRGCN, PostEnsembleDynamicRGCN
from models.PostBiDynamicRGCN import ImputeBiDynamicRGCN, PostBiDynamicRGCN, PostEnsembleBiDynamicRGCN
from models.PostSelfAttentionRGCN import PostBiSelfAttentionRGCN, PostSelfAttentionRGCN

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
              "Simple": SimplE,
              "Static": Static,
              "DE": DiachronicEmbedding,
              "Hyte": Hyte,
              "SRGCN": StaticRGCN,
              "GRRGCN": DynamicRGCN,
              "RRGCN": DynamicRGCN,
              "SARGCN": SelfAttentionRGCN,
              "BiSARGCN": BiSelfAttentionRGCN,
              "BiGRRGCN": BiDynamicRGCN,
              "BiRRGCN": BiDynamicRGCN,
              "Aggregator": Aggregator
              }[args.module]

    if module == BiDynamicRGCN:
        if args.post_aggregation:
            module = PostBiDynamicRGCN
        elif args.post_ensemble:
            module = PostEnsembleBiDynamicRGCN
        elif args.impute:
            module = ImputeBiDynamicRGCN


    elif module == DynamicRGCN:
        if args.post_aggregation:
            module = PostDynamicRGCN
        if args.post_ensemble:
            module = PostEnsembleDynamicRGCN
        elif args.impute:
            module = ImputeDynamicRGCN

    elif module == BiSelfAttentionRGCN:
        if args.post_aggregation:
            module = PostBiSelfAttentionRGCN
    elif module == SelfAttentionRGCN:
        if args.post_aggregation:
            module = PostSelfAttentionRGCN

    # import pdb; pdb.set_trace()
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
        name="{}-{}-{}-{}-{}-{}time-embed-{}only-last-layer-{}-dropout-{}-learnable-{}-ensemble-{}-impute".format(args.module, args.dataset.split('/')[-1], args.score_function,
                                     args.train_seq_len, args.inv_temperature, '' if args.use_time_embedding else 'no-', '' if args.rec_only_last_layer else 'not-',
                                                                                "edge" if args.edge_dropout else "random" if args.random_dropout else "no",
                                                                                "lambda" if args.learnable_lambda else "not",
                                                                                "embedding" if args.post_aggregation else "score" if args.post_ensemble else "no",
                                                                                "with" if args.impute else "without"),
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

    if args.resume:
        assert args.model_name, args.version
        tt_logger = MyTestTubeLogger(
            save_dir='experiments',
            name=args.model_name,
            debug=False,
            version=args.version  # An existing version with a saved checkpoint
        )

    tt_logger.log_hyperparams(args)
    tt_logger.save()
    # most basic trainer, uses good defaults

    trainer = Trainer(logger=tt_logger, gpus=args.n_gpu,
                      gradient_clip_val=args.gradient_clip_val,
                      max_nb_epochs=args.max_nb_epochs,
                      fast_dev_run=args.fast_dev_run,
                      distributed_backend=args.distributed_backend,
                      nb_sanity_val_steps=1 if args.debug else 1,
                      early_stop_callback=early_stop_callback,
                      train_percent_check=0.1 if args.debug else 1.0,
                      checkpoint_callback=checkpoint_callback
                      )

    trainer.fit(model)
    trainer.test()


