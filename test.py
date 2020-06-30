from utils.dataset import *
from utils.args import process_args
from previous.TKG_VRE import TKG_VAE
from baselines.Static import Static
from baselines.Simple import SimplE
from baselines.Hyte import Hyte
from baselines.DiachronicEmbedding import DiachronicEmbedding
from baselines.StaticRGCN import StaticRGCN
from models.DynamicRGCN import DynamicRGCN
import glob
import json
from pytorch_lightning import Trainer
import torch.distributed as dist
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.root_module.root_module import LightningModule
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from models.BiDynamicRGCN import BiDynamicRGCN
from models.SelfAttentionRGCN import SelfAttentionRGCN
from models.BiSelfAttentionRGCN import BiSelfAttentionRGCN
from models.aggregator import Aggregator
from models.PostDynamicRGCN import ImputeDynamicRGCN, PostDynamicRGCN, PostEnsembleDynamicRGCN
from models.PostBiDynamicRGCN import ImputeBiDynamicRGCN, PostBiDynamicRGCN, PostEnsembleBiDynamicRGCN
from models.PostSelfAttentionRGCN import PostBiSelfAttentionRGCN, PostSelfAttentionRGCN
import os.path

class MyTrainer(Trainer):
    def test(self, model=None):
        return self.__single_gpu_train(model)

    def __configure_checkpoint_callback(self):
        """
        Weight path set in this priority:
        Checkpoint_callback's path (if passed in).
        User provided weights_saved_path
        Otherwise use os.getcwd()
        """
        if self.checkpoint_callback is None:
            # init a default one
            if isinstance(self.logger, TestTubeLogger):
                ckpt_path = '{}/{}/version_{}/{}'.format(
                    self.default_save_path,
                    self.logger.experiment.name,
                    self.logger.experiment.version,
                    'checkpoints')
            else:
                ckpt_path = self.default_save_path

            self.checkpoint_callback = ModelCheckpoint(
                filepath=ckpt_path
            )

        # set the path for the callbacks
        self.checkpoint_callback.save_function = self.save_checkpoint

        # if checkpoint callback used, then override the weights path
        self.weights_save_path = self.checkpoint_callback.filepath

        # if weights_save_path is still none here, set to current working dir
        if self.weights_save_path is None:
            self.weights_save_path = self.default_save_path

    def __layout_bookeeping(self):

        # determine number of training batches
        self.nb_training_batches = len(self.get_train_dataloader())
        self.nb_training_batches = int(self.nb_training_batches * self.train_percent_check)

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.get_val_dataloaders() is not None:
            self.nb_val_batches = sum(len(dataloader) for dataloader in self.get_val_dataloaders())
            self.nb_val_batches = int(self.nb_val_batches * self.val_percent_check)
            self.nb_val_batches = max(1, self.nb_val_batches)

        # determine number of test batches
        if self.get_test_dataloaders() is not None:
            self.nb_test_batches = sum(
                len(dataloader) for dataloader in self.get_test_dataloaders()
            )
            self.nb_test_batches = int(self.nb_test_batches * self.test_percent_check)
            self.nb_test_batches = max(1, self.nb_test_batches)

        # determine when to check validation
        self.val_check_batch = int(self.nb_training_batches * self.val_check_interval)
        self.val_check_batch = max(1, self.val_check_batch)

    def __single_gpu_train(self, model):
        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        model.cuda(self.root_gpu)

        return self.__run_pretrain_routine(model)

    def __run_pretrain_routine(self, model=None):
        """
        Sanity check a few things before starting actual training
        :param model:
        :return:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        ref_model.on_gpu = self.on_gpu
        ref_model.single_gpu = self.single_gpu
        ref_model.use_dp = self.use_dp
        ref_model.use_ddp = self.use_ddp
        ref_model.use_ddp2 = self.use_ddp2
        ref_model.use_amp = self.use_amp
        ref_model.testing = self.testing

        # link up experiment object
        if self.logger is not None:
            ref_model.logger = self.logger

            # save exp to get started
            if hasattr(ref_model, "hparams"):
                self.logger.log_hyperparams(ref_model.hparams)

            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            dist.barrier()

        # set up checkpoint callback
        self.__configure_checkpoint_callback()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # init training constants
        self.__layout_bookeeping()

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        return self.__run_evaluation(test=True)

    def __is_overriden(self, f_name):
        model = self.__get_model()
        super_object = LightningModule

        # when code pointers are different, it was overriden
        is_overriden = getattr(model, f_name).__code__ is not getattr(super_object, f_name).__code__
        return is_overriden

    def __get_model(self):
        return self.model.module if self.data_parallel else self.model

    def __run_evaluation(self, test=False):
        # when testing make sure user defined a test step
        can_run_test_step = False
        if test:
            can_run_test_step = self.__is_overriden('test_step') and self.__is_overriden('test_end')

        # validate only if model has validation_step defined
        # test only if test_step or validation_step are defined
        run_val_step = self.__is_overriden('validation_step')

        if run_val_step or can_run_test_step:

            # hook
            model = self.__get_model()
            model.on_pre_performance_check()

            # select dataloaders
            dataloaders = self.get_val_dataloaders()
            max_batches = self.nb_val_batches

            # calculate max batches to use
            if test:
                dataloaders = self.get_test_dataloaders()
                max_batches = self.nb_test_batches

            # cap max batches to 1 when using fast_dev_run
            if self.fast_dev_run:
                max_batches = 1

            # run evaluation
            return self.evaluate(self.model, dataloaders, max_batches, test)

    def __evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)

        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if type(self.data_parallel_device_ids) is list:
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # CPU
        if test:
            # pdb.set_trace()
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """
         :param dataloaders: list of PT dataloaders
         :param max_batches: Scalar
         :param test: boolean
         :return:
         """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run training
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):

                if batch is None:  # pragma: no cover
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.__evaluation_forward(model,
                                                   batch,
                                                   batch_idx,
                                                   dataloader_idx,
                                                   test)

                # track outputs for collation
                dl_outputs.append(output)

            outputs.append(dl_outputs)

        eval_results = {}

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        model = self.__get_model()
        if test and self.__is_overriden('test_end'):
            eval_results = model.test_end(outputs)
        elif self.__is_overriden('validation_end'):
            eval_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)
        # pdb.set_trace()
        return eval_results


def get_batch_graph_list(t_list, seq_len, graph_dict):
    times = list(graph_dict.keys())
    # time_unit = times[1] - times[0]  # compute time unit
    time_list = []

    t_list = t_list.sort(descending=True)[0]
    g_list = []
    # s_lst = [t/15 for t in times]
    # print(s_lst)
    for tim in t_list:
        # length = int(tim / time_unit) + 1
        # cur_seq_len = seq_len if seq_len <= length else length
        length = times.index(tim) + 1
        time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
        time_list.append(([None] * (seq_len - len(time_seq))) + time_seq)
        g_list.append(([None] * (seq_len - len(time_seq))) + [graph_dict[t] for t in time_seq])
    t_batched_list = [list(x) for x in zip(*time_list)]
    g_batched_list = [list(x) for x in zip(*g_list)]
    return g_batched_list, t_batched_list


def get_predictions(batch_times, all_ranks, model):
    predictions = []
    for t_list, ranks in zip(batch_times, all_ranks):
        g_batched_list, t_batched_list = get_batch_graph_list(t_list, 1, model.graph_dict_test)
        batch_five_tuples = []
        six_tuples = []
        for g, t in zip(g_batched_list[-1], t_batched_list[-1]):
            triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1).tolist()
            # pdb.set_trace()
            # 's' = 'head', predicting s given (o, r)
            # 'o' = 'tail', predicting o given (s, r)
            five_tup_s = [[g.ids[x[0]], x[1], g.ids[x[2]], t, 's'] for x in triples]
            five_tup_o = [[g.ids[x[0]], x[1], g.ids[x[2]], t, 'o'] for x in triples]
            batch_five_tuples.extend(five_tup_s)
            batch_five_tuples.extend(five_tup_o)
        for five_tup, rank in zip(batch_five_tuples, ranks.tolist()):
            six_tuples.append(five_tup + [rank])
        predictions.extend(six_tuples)
    return predictions


def inference():
    # if os.path.isfile(prediction_file):
    #     with open(prediction_file, 'rb') as filehandle:
    #         # read the data as binary data stream
    #         predictions = pickle.load(filehandle)
    # else:
    # import pdb; pdb.set_trace()

    if not args.module == "Aggregator":
        experiment_path = args.checkpoint_path
        checkpoint_path = glob.glob(os.path.join(experiment_path, "checkpoints", "*.ckpt"))[0]
        # checkpoint_path = experiment_path
        joined = "-".join(experiment_path.split('/')[1:])
        prediction_file = os.path.join(experiment_path, joined + "-predictions.pk")
        print(prediction_file)
        config_path = os.path.join(experiment_path, "config.json")
        args_json = json.load(open(config_path))
        args.__dict__.update(dict(args_json))

        # if os.path.isfile(prediction_file):
        #     return
    args.use_VAE = False
    use_cuda = args.use_cuda = args.n_gpu >= 0 and torch.cuda.is_available()
    num_ents, num_rels = get_total_number(args.dataset, 'stat.txt')

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

    if not args.module == "Aggregator":
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # model.test_seq_len = args.test_seq_len
        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)
    trainer = MyTrainer(gpus=0 if not use_cuda else 1)
    results = trainer.test(model)
    all_ranks = results['all_ranks']
    batch_times = results['batch_times']
    predictions = get_predictions(batch_times, all_ranks, model)

    with open(prediction_file, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(predictions, filehandle)
    return predictions


if __name__ == '__main__':
    args = process_args()
    torch.manual_seed(args.seed)
    predictions = inference()
    # train_entity_freq, train_ent_rel_freq, train_ent_pair_freq = get_train_data_freq()
    # per_entity_ranks = calc_per_entity_prediction(predictions)
    # hist_freq_entity_mrr(train_entity_freq, per_entity_ranks)
    # hist_freq_entity_pair_mrr(train_ent_pair_freq, per_entity_ranks)
    # hist_freq_entity_rel_mrr(train_ent_rel_freq, per_entity_ranks)