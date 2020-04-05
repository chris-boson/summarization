import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from common.metrics import Metrics
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup

from common.logger import logger


class SummarizationModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, dataset_class: torch.utils.data.Dataset):
        super().__init__()
        self.hparams = hparams
        self.dataset_class = dataset_class
        self.clean_hparams()
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.metrics = Metrics()

    @staticmethod
    def add_args(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_batch_size', default=2, type=int, help="Train batch size.")
        parser.add_argument('--eval_batch_size', default=1, type=int, help="Eval batch size.")
        parser.add_argument('--lr', default=5e-5, type=float, help="Learning Rate")
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--encoder", default=None, type=str, help="Encoder architecture.")
        parser.add_argument("--decoder", default=None, type=str, help="Decoder architecture.")
        parser.add_argument("--num_beams", default=None, type=int, help="Width of beam search.")
        parser.add_argument("--max_length", default=40, type=int, help="Max number of tokens of generated summaries.")
        parser.add_argument("--repetition_penalty", default=3.0, type=float, help="Penalize repetition. More than 1.0 -> less repetition.")
        parser = SummarizationModel.add_dataset_args(parser)
        return parser

    @staticmethod
    def add_dataset_args(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_tokens', default=1024, type=int, help="Maximum number of input tokens.")
        parser.add_argument('--max_documents', default=None, type=int, help="Maximum number of documents.")
        return parser

    def clean_hparams(self):
        if 'kwargs' in self.hparams.__dict__:
            del self.hparams.__dict__['kwargs']
        # Related to the issue here: https://github.com/PyTorchLightning/pytorch-lightning/pull/1128
        self.hparams.__dict__ = {
            k: v for k, v in self.hparams.__dict__.items()
            if v is not None and not isinstance(v, list)
        }
        logger.info(json.dumps(self.hparams.__dict__, indent=4))

    def get_datasets(self):
        if self.hparams.dataset == 'tifu':
            assert self.hparams.test_percentage < 0.33
            self.dataset = self.dataset_class(self.hparams, self.encoder_tokenizer)
            test_len = int(self.hparams.test_percentage*len(self.dataset))
            lengths = [len(self.dataset) - 2*test_len, test_len, test_len]
            self.datasets = torch.utils.data.random_split(self.dataset, lengths)
        elif self.hparams.dataset == 'cnn_dm':
            self.datasets = [
                self.dataset_class(self.hparams, self.encoder_tokenizer, type_path=i)
                for i in ['train', 'val', 'test']
            ]
        logger.info("Documents train: %s, val: %s, test: %s" %
            tuple(len(dataset) for dataset in self.datasets))

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            num_workers=4,
            collate_fn=self.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=4,
            collate_fn=self.collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=4,
            collate_fn=self.collate
        )

    def collate(self, batch):
        inputs = [elem[0] for elem in batch]
        labels = [elem[1] for elem in batch]
        pad_token_id = self.encoder_tokenizer.pad_token_id
        # pad_token_id = 0
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
        inputs_mask = (inputs_padded != 0).int()
        labels_mask = (labels_padded != 0).int()
        return [inputs_padded, labels_padded, inputs_mask, labels_mask]

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

        if 'max_steps' in self.hparams.__dict__:
            t_total = self.hparams.max_steps
        else:
            t_total = len(self.train_dataloader()) * self.hparams.max_epochs
            t_total /= self.hparams.accumulate_grad_batches
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        return [optimizer], [self.scheduler]

    def decode(self, ids):
        return self.decoder_tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, batch):
        return [
            {
                'prediction': self.decode(batch['preds'][i]),
                'target': self.decode(batch['target'][i])
            } for i in range(len(batch['preds']))
        ]

    def calculate_metrics(self, output):
        all_predictions = [obj["prediction"] for obj in output]
        all_targets = [obj["target"] for obj in output]

        metric_scores = self.metrics.score(all_predictions, all_targets)
        logger.info(json.dumps(metric_scores, indent=4))
        return metric_scores
