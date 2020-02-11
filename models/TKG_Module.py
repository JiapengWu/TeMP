import numpy as np
from utils.scores import *
import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import TimeDataset
from argparse import Namespace
from utils.CorrptTriples import CorruptTriples
import torch.nn.functional as F
from utils.utils import filter_none
from utils.evaluation import EvaluationFilter
from utils.utils import cuda
from utils.dataset import load_quadruples


class TKG_Module(LightningModule):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TKG_Module, self).__init__()
        self.args = self.hparams = args
        self.graph_dict_train = graph_dict_train
        self.graph_dict_val = graph_dict_val
        self.graph_dict_test = graph_dict_test
        self.total_time = list(graph_dict_train.keys())
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.use_cuda = args.use_cuda
        self.num_pos_facts = args.num_pos_facts
        self.negative_rate = args.negative_rate
        self.calc_score = {'distmult': distmult, 'complex': complex, 'transE': transE}[args.score_function]
        self.build_model()
        self.set_extra_vars()
        self.corrupter = CorruptTriples(self.args, graph_dict_train)
        self.evaluater = EvaluationFilter(args, self.calc_score, graph_dict_train, graph_dict_val, graph_dict_test)

    def training_step(self, batch_time, batch_idx):
        loss = self.forward(batch_time)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self.logger.experiment.log(output)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch_time, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        ranks, loss = self.evaluate(batch_time)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        log_output = OrderedDict({
            'mean_ranks': ranks.float().mean().item(),
            'val_loss': loss,
        })
        output = OrderedDict({
            'ranks': ranks,
            'val_loss': loss,
        })
        self.logger.experiment.log(log_output)
        return output

    def validation_end(self, outputs):
        avg_val_loss = np.mean([x['val_loss'] for x in outputs])
        all_ranks = torch.cat([x['ranks'] for x in outputs])
        mrr, hit_1, hit_3, hit_10 = self.get_metrics(all_ranks)

        return {'mrr': mrr,
                'avg_val_loss': avg_val_loss,
                'hit_10': hit_10,
                'hit_3': hit_3,
                'hit_1': hit_1
                }

    def test_step(self, batch_time, batch_idx):
        ranks, loss = self.evaluate(batch_time, val=False)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        log_output = OrderedDict({
            'mean_ranks': ranks.float().mean().item(),
            'test_loss': loss,
        })

        output = OrderedDict({
            'ranks': ranks,
            'test_loss': loss,
        })
        self.logger.experiment.log(log_output)

        return output

    def test_end(self, outputs):
        avg_test_loss = np.mean([x['test_loss'] for x in outputs])
        all_ranks = torch.cat([x['ranks'] for x in outputs])
        mrr, hit_1, hit_3, hit_10 = self.get_metrics(all_ranks)

        test_result = {'mrr': mrr,
                        'avg_test_loss': avg_test_loss,
                        'hit_10': hit_10,
                        'hit_3': hit_3,
                        'hit_1': hit_1
                        }

        print(test_result)
        print()
        return test_result

    def set_extra_vars(self):
        if not self.args.dataset_dir == 'extrapolation':
            return
        self.graph_dict_total = {**self.graph_dict_train, **self.graph_dict_val, **self.graph_dict_test}
        _, self.train_times = load_quadruples(self.args.dataset, 'train.txt')
        _, self.test_times = load_quadruples(self.args.dataset, 'test.txt')
        if self.args.dataset == 'extrapolation/icews14':
            _, self.valid_times = load_quadruples(self.args.dataset, 'test.txt')
            _, self.total_times = load_quadruples(self.args.dataset, 'train.txt', 'test.txt')
        else:
            _, self.valid_times = load_quadruples(self.args.dataset, 'valid.txt')
            _, self.total_times = load_quadruples(self.args.dataset, 'train.txt', 'valid.txt','test.txt')
        self.time_unit = self.total_times[1] - self.total_times[0]

    def get_metrics(self, ranks):
        mrr = torch.mean(1.0 / ranks.float())
        hit_1 = torch.mean((ranks <= 1).float())
        hit_3 = torch.mean((ranks <= 3).float())
        hit_10 = torch.mean((ranks <= 10).float())
        return mrr, hit_1, hit_3, hit_10

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0.0001)
        return optimizer

    def _dataloader(self, times):
        # when using multi-node (ddp) we need to add the  datasampler
        dataset = TimeDataset(times)
        batch_size = self.args.batch_size
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        return self._dataloader(self.total_time)

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloader(self.total_time)

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.total_time)

    # TODO: fix training so that all nodes including isolated ones are considered in the evaluation
    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, all_embeds_g, corrupt_tail=True):
        r = self.rel_embeds[triplets[:, 1]]

        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            neg_o = all_embeds_g[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = all_embeds_g[neg_samples]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')
        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def link_classification_loss(self, ent_embed, rel_embeds, triplets, labels):
        # triplets is a list of extrapolation samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = rel_embeds[triplets[:, 1]]
        o = ent_embed[triplets[:, 2]]
        score = self.calc_score(s, r, o)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

    def get_pooled_facts(self, ent_embed, triples):
        s = ent_embed[triples[:, 0]]
        r = self.rel_embeds[triples[:, 1]]
        o = ent_embed[triples[:, 2]]
        pos_facts = torch.cat([s, r, o], dim=1)
        return torch.max(pos_facts, dim=0)[0]

    def get_batch_graph_list(self, t_list, seq_len, graph_dict):
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]  # compute time unit
        time_list = []

        t_list = t_list.sort(descending=True)[0]
        g_list = []
        for tim in t_list:
            length = int(tim / time_unit) + 1
            # cur_seq_len = seq_len if seq_len <= length else length
            time_seq = times[length - seq_len:length] if seq_len <= length else times[:length]
            time_list.append(([None] * (seq_len - len(time_seq))) + time_seq)
            g_list.append(([None] * (seq_len - len(time_seq))) + [graph_dict[t] for t in time_seq])

        t_batched_list = [list(x) for x in zip(*time_list)]
        g_batched_list = [list(x) for x in zip(*g_list)]
        return g_batched_list, t_batched_list

    # TODO: fix eval so that all nodes including isolated ones are considered in the evaluation
    def calc_metrics(self, g_list, t_list):
        mrrs, hit_1s, hit_3s, hit_10s, losses = [], [], [], [], []
        ranks = []
        for g, t in zip(g_list, t_list):

            ent_embed = self.get_per_graph_ent_embeds(t, g)
            all_embeds_g = self.get_all_embeds_Gt(t)
            index_sample = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            label = torch.ones(index_sample.shape[0])
            if self.use_cuda:
                index_sample = cuda(index_sample)
                label = cuda(label)

            if index_sample.shape[0] == 0: continue

            rank = self.evaluater.calc_metrics_single_graph(ent_embed, self.rel_embeds, all_embeds_g, index_sample, g, t)
            loss = self.link_classification_loss(ent_embed, self.rel_embeds, index_sample, label)
            ranks.append(rank)
            losses.append(loss.item())
        ranks = torch.cat(ranks)

        return ranks, np.mean(losses)
