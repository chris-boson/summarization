import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from trainer.models.base import SummarizationModel


class GPT2LMSummarizer(SummarizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.get_tokenizers()
        self.get_datasets()

    def get_tokenizers(self):
        self.encoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

    def training_step(self, batch, batch_nb):
        input_ids, _, input_mask, _ = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask
        )
        loss, logits = outputs[:2]
        log = {'train/loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_nb):
        input_ids, _, input_mask, _ = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask
        )
        loss, logits = outputs[:2]
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)
