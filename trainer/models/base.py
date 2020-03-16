import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup


class SummarizationModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, dataset_class: torch.utils.data.Dataset):
        super().__init__()
        self.hparams = hparams
        self.dataset_class = dataset_class
        self.clean_hparams()

    @staticmethod
    def add_args(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_batch_size', default=2, type=int, help="Train batch size.")
        parser.add_argument('--eval_batch_size', default=1, type=int, help="Eval batch size.")
        parser.add_argument('--lr', default=5e-5, type=float, help="Learning Rate")
        parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--encoder", default='gpt2', type=str, help="Encoder architecture.")
        parser.add_argument("--decoder", default='gpt2', type=str, help="Decoder architecture.")
        parser = SummarizationModel.add_dataset_args(parser)
        return parser

    @staticmethod
    def add_dataset_args(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_tokens', default=1024, type=int, help="Maximum number of input tokens.")
        parser.add_argument('--max_documents', default=None, type=int, help="Maximum number of documents.")
        return parser

    def clean_hparams(self):
        del self.hparams.__dict__['kwargs']
        # Related to the issue here: https://github.com/PyTorchLightning/pytorch-lightning/pull/1128
        self.hparams.__dict__ = {
            k: v for k, v in self.hparams.__dict__.items()
            if v is not None and not isinstance(v, list)
        }
        print(json.dumps(self.hparams.__dict__, indent=4))

    def get_datasets(self):
        if self.hparams.dataset == 'tifu':
            assert self.hparams.test_percentage < 0.33
            self.dataset = self.dataset_class(self.hparams, self.encoder_tokenizer)
            test_len = int(self.hparams.test_percentage*len(self.dataset))
            lengths = [len(self.dataset) - 2*test_len, test_len, test_len]
            print("Documents train: %s, val: %s, test: %s" % tuple(lengths))
            self.datasets = torch.utils.data.random_split(self.dataset, lengths)
        elif self.hparams.dataset == 'cnn_dm':
            self.datasets = [
                self.dataset_class(self.hparams, self.encoder_tokenizer, type_path=i)
                for i in ['train', 'val', 'test']
            ]

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            num_workers=1,
            collate_fn=self.dataset.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=1,
            collate_fn=self.dataset.collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=1,
            collate_fn=self.dataset.collate
        )

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

    # Work around for ddp distribution strategy: https://github.com/PyTorchLightning/pytorch-lightning/issues/538
    def configure_ddp(self, model, device_ids):
        """
        Configure to use a single GPU set on local rank.

        Must return model.
        :param model:
        :param device_ids:
        :return: DDP wrapped model
        """
        device_id = f"cuda:{os.environ['LOCAL_RANK']}"

        model = pl.overrides.data_parallel.LightningDistributedDataParallel(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=True,
        )

        return model

    def init_ddp_connection(self, proc_rank, world_size):
        """
        Connect all procs in the world using the env:// init
        Use the first node as the root address
        """

        import torch.distributed as dist

        dist.init_process_group("nccl", init_method="env://")
        FIXED_SEED = 0

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        # TODO(jeffling): I'm pretty sure we need to set other seeds as well?
        print(f"Setting torch manual seed to {FIXED_SEED} for DDP.")
        torch.manual_seed(FIXED_SEED)
