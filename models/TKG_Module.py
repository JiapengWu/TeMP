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
        return self._dataloader(self.train_times)
        # return self._dataloader(self.valid_times)

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloader(self.test_times)

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