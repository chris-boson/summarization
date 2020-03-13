import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from trainer.dataset import TIFUDataset
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup


class SummarizationModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.dataset_class = TIFUDataset
        self.clean_hparams()
        self.get_datasets()

    @staticmethod
    def add_args(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TIFUDataset.add_args(parser)
        parser.add_argument('--train_batch_size', default=2, type=int, help="Train batch size.")
        parser.add_argument('--lr', default=0.02, type=float, help="Learning Rate")
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        return parser

    def clean_hparams(self):
        # Related to the issue here: https://github.com/PyTorchLightning/pytorch-lightning/pull/1128
        for k, v in self.hparams.__dict__.items():
            if v is None or isinstance(v, list): self.hparams.__dict__[k] = 'None'
        del self.hparams.__dict__['kwargs']
        print(json.dumps(self.hparams.__dict__, indent=4))

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

    def training_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, label_mask = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=label_ids
        )
        loss, logits = outputs[:2]
        log = {'train/loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, label_mask = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            labels=label_ids
        )
        loss, logits = outputs[:2]
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

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
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            eps=self.hparams.adam_epsilon
        )

        if self.hparams.max_steps != 'None':
            t_total = self.hparams.max_steps
        else:
            t_total = len(self.train_dataloader()) * int(self.hparams.max_epochs)
            t_total /= self.hparams.accumulate_grad_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        return [optimizer], [scheduler]

    def get_datasets(self):
        dataset = self.dataset_class(self.hparams)
        test_len = int(self.hparams.test_percentage*len(dataset))
        lengths = [len(dataset) - test_len, test_len]
        self.datasets = torch.utils.data.random_split(dataset, lengths)

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            num_workers=1,
            collate_fn=self.dataset_class.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            shuffle=False,
            batch_size=self.hparams.train_batch_size,
            num_workers=1,
            collate_fn=self.dataset_class.collate
        )
