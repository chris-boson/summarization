import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import PreTrainedEncoderDecoder
from transformers.modeling_bart import BartForConditionalGeneration
from transformers import BartTokenizer
from trainer.models.base import SummarizationModel


class BartSummarizer(SummarizationModel):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__(hparams)
        self.model = BartForConditionalGeneration.from_pretrained(
            'bart-large', output_past=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.get_tokenizers()
        self.get_datasets()

    def get_tokenizers(self):
        self.encoder_tokenizer = BartTokenizer.from_pretrained('bart-large')
        self.decoder_tokenizer = BartTokenizer.from_pretrained('bart-large')

    def forward(self, input_ids, attention_mask, label_ids, label_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=label_ids#,
            # decoder_attention_mask=label_mask
        )

    def _step(self, batch):
        input_ids, label_ids, input_mask, label_mask = batch
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=input_mask,
            label_ids=label_ids,
            label_mask=label_mask
        )
        out = outputs[0]

        logits = F.log_softmax(out, dim=-1)
        y = label_ids
        norm = (y != self.encoder_tokenizer.pad_token_id).data.sum()

        targets = y.clone()
        targets[y == self.encoder_tokenizer.pad_token_id] = -100
        loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1)) / norm
        return loss

    def training_step(self, batch, batch_nb):
        loss = self._step(batch)
        log = {'train/loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_nb):
        loss = self._step(batch)
        input_ids, label_ids, input_mask, _ = batch
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=input_mask,
            # num_beams=3,
            max_length=40,
            repetition_penalty=3.0,
        )
        preds = [
            self.decoder_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [
            self.decoder_tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in label_ids
        ]
        return {"val_loss": loss, "preds": preds, "target": target}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)