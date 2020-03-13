import argparse
import json
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from trainer.dataset import TIFUDataset
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup


class SummarizationModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        for k, v in hparams.__dict__.items():
            if v is None or isinstance(v, list): hparams.__dict__[k] = 'None'
        del hparams.__dict__['kwargs']
        print(json.dumps(hparams.__dict__, indent=4))
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

        if self.hparams.max_steps != 'None':# is not None:
            t_total = self.hparams.max_steps
        else:
            t_total = len(self.train_dataloader()) * self.hparams.max_epochs
            t_total /= self.hparams.accumulate_grad_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            TIFUDataset(self.hparams.input_dir),
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            num_workers=1,
            collate_fn=TIFUDataset.collate
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
        parser.add_argument('--train_batch_size', default=2, type=int, help="Train batch size.")
        parser.add_argument('--lr', default=0.02, type=float, help="Learning Rate")
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

        return parser
