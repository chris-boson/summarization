import os
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from trainer.dataset import TIFUDataset
from transformers import GPT2LMHeadModel

class SummarizationModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        for k, v in hparams.__dict__.items():
            if v is None or isinstance(v, list): hparams.__dict__[k] = 'None'
        print(hparams.__dict__)
        self.hparams = hparams
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

    def training_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, label_mask = batch
        # print(input_ids.shape, input_mask.shape, label_ids.shape, label_mask.shape)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=label_ids
        )
        loss, logits = outputs[:2]
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'val_loss': F.cross_entropy(y_hat, y)}

    # def validation_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}

    # def test_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        input_path = os.path.join(
            self.hparams.input_dir, 'tifu_all_tokenized_and_filtered.json'
        )
        return DataLoader(
            TIFUDataset(input_path),
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            num_workers=1,
            collate_fn=TIFUDataset.my_collate
        )

    # def val_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(self.hparams.input_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(self.hparams.input_dir, train=False, download=True, transform=transforms.ToTensor()), batch_size=32)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.args
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--lr', default=0.02, type=float, help="Learning Rate")
        parser.add_argument('--train_batch_size', default=2, type=int, help="Train batch size.")
        return parser