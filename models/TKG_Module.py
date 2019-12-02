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


class TKG_Module(LightningModule):
    def __init__(self, args, num_ents, num_rels, graph_dict_total, train_times, valid_times, test_times):
        super(TKG_Module, self).__init__()
        self.args = self.hparams = args
        self.graph_dict_total = graph_dict_total
        self.train_times = train_times
        self.valid_times = valid_times
        self.test_times = test_times
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.use_cuda = args.use_cuda
        self.num_pos_facts = args.num_pos_facts
        self.negative_rate = args.negative_rate
        self.calc_score = {'distmult': distmult, 'complex': complex}[args.score_function]
        self.build_model()
        self.corrupter = CorruptTriples(self.args, graph_dict_total)

    def training_step(self, batch_time, batch_idx):
        reconstruct_loss, kld_loss = self.forward(batch_time)
        loss = reconstruct_loss + kld_loss

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'train_reconstruction_loss': reconstruct_loss,
            'train_KLD_loss': kld_loss,
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        if type(kld_loss) != torch.Tensor:
            del output['train_KLD_loss']

        self.logger.experiment.log(output)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch_time, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        mrrs, hit_1s, hit_3s, hit_10s, loss, acc_reconstruct_loss = self.evaluate(batch_time)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        output = OrderedDict({
            'MRR': mrrs,
            'Hit_1': hit_1s,
            'Hit_3': hit_3s,
            'Hit_10': hit_10s,
            'val_loss': loss,
            'val_acc_reconstruct_loss': acc_reconstruct_loss
        })
        if type(acc_reconstruct_loss) != torch.Tensor:
            del output['val_acc_reconstruct_loss']
        # print(output)
        self.logger.experiment.log(output)
        return output

    def validation_end(self, outputs):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mrrs = np.mean([x['MRR'] for x in outputs])
        avg_val_loss = np.mean([x['val_loss'] for x in outputs])
        return {'avg_mrr': avg_mrrs, 'avg_val_loss': avg_val_loss}

    def test_step(self, batch_time, batch_idx):
        mrrs, hit_1s, hit_3s, hit_10s, loss, acc_reconstruct_loss = self.evaluate(batch_time)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        output = OrderedDict({
            'MRR': mrrs,
            'Hit_1': hit_1s,
            'Hit_3': hit_3s,
            'Hit_10': hit_10s,
            'test_loss': loss,
            'test_acc_reconstruct_loss': acc_reconstruct_loss
        })
        self.logger.experiment.log(output)

        if type(acc_reconstruct_loss) != torch.Tensor:
            del output['val_acc_reconstruct_loss']
        return output

    def test_end(self, outputs):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mrrs = np.mean([x['MRR'] for x in outputs])
        avg_val_loss = np.mean([x['test_loss'] for x in outputs])
        return {'avg_mrr': avg_mrrs, 'avg_test_loss': avg_val_loss}

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
        return self._dataloader(self.train_times)

    @pl.data_loader
    def val_dataloader(self):
        # return self._dataloader(self.train_times)
        return self._dataloader(self.valid_times)

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.test_times)

    def train_link_prediction(self, ent_embed, triplets, neg_samples, labels, corrupt_tail=True):
        if corrupt_tail:
            s = ent_embed[triplets[:, 0]]
            r = self.rel_embeds[triplets[:, 1]]
            neg_o = ent_embed[neg_samples]
            score = self.calc_score(s, r, neg_o, mode='tail')
        else:
            neg_s = ent_embed[neg_samples]
            r = self.rel_embeds[triplets[:, 1]]
            o = ent_embed[triplets[:, 2]]
            score = self.calc_score(neg_s, r, o, mode='head')
        # pdb.set_trace()
        predict_loss = F.cross_entropy(score, labels)
        return predict_loss

    def link_classification_loss(self, ent_embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        s = ent_embed[triplets[:, 0]]
        r = self.rel_embeds[triplets[:, 1]]
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

    def get_val_vars(self, g_batched_list, t, h):
        g_batched_list_t = filter_none(g_batched_list[t])
        bsz = len(g_batched_list_t)
        triplets, labels = self.corrupter.sample_labels_val(g_batched_list_t)
        cur_h = h[-1][:bsz]  # bsz, hidden_size
        # run RGCN on graph to get encoded ent_embeddings and rel_embeddings in G_t
        node_sizes = [len(g.nodes()) for g in g_batched_list_t]
        return g_batched_list_t, bsz, cur_h, triplets, labels, node_sizes

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, num_ents, num_rels, graph_dict_train, graph_dict_dev, graph_dict_test, train_times, valid_times, test_times):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        try:
            ckpt_hparams = checkpoint['hparams']
        except KeyError:
            raise IOError(
                "Checkpoint does not contain hyperparameters. Are your model hyperparameters stored"
                "in self.hparams?"
            )
        hparams = Namespace(**ckpt_hparams)

        # load the state_dict on the model automatically
        model = cls(hparams, checkpoint_path, num_ents, num_rels, graph_dict_train, graph_dict_dev, graph_dict_test, train_times, valid_times, test_times)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model