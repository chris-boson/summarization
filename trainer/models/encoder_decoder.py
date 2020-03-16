import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import PreTrainedEncoderDecoder
from transformers import GPT2Tokenizer, BertTokenizer
from trainer.models.base import SummarizationModel


class EncoderDecoderSummarizer(SummarizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PreTrainedEncoderDecoder.from_pretrained(
            self.hparams.encoder, self.hparams.decoder
        )
        self.get_tokenizers()
        self.get_datasets()

    def get_tokenizers(self):
        tokenizer_dict = {
            'gpt2': GPT2Tokenizer,
            'bert-base-uncased': BertTokenizer
        }
        self.encoder_tokenizer = tokenizer_dict[self.hparams.encoder].from_pretrained(self.hparams.encoder)
        self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
        self.decoder_tokenizer = tokenizer_dict[self.hparams.decoder].from_pretrained(self.hparams.decoder)

    def forward(self, input_ids, attention_mask, label_ids, label_mask):
        return self.model(
            encoder_input_ids=input_ids,
            # encoder_attention_mask=attention_mask,
            decoder_input_ids=label_ids#,
            # decoder_attention_mask=label_mask
        )

    def training_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, label_mask = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            label_ids=label_ids,
            label_mask=label_mask
        )
        loss, logits = outputs[:2]
        log = {'train/loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, label_mask = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            label_ids=label_ids,
            label_mask=label_mask
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