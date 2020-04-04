
import json
import os
import torch
from torch.nn import functional as F

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from trainer.models.base import SummarizationModel
from trainer.logger import get_logger

logger = get_logger()
class ConditionalGenerationSummarizer(SummarizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.get_model()
        self.get_datasets()

    def get_model(self):
        model_dict = {
            'bart' : (BartTokenizer, BartForConditionalGeneration),
            't5': (T5Tokenizer, T5ForConditionalGeneration)
        }
        assert any(model_type in self.hparams.encoder for model_type in model_dict.keys())
        for k, v in model_dict.items():
            if k in self.hparams.encoder:
                self.encoder_tokenizer = v[0].from_pretrained(self.hparams.encoder)
                self.decoder_tokenizer = v[0].from_pretrained(self.hparams.decoder)
                self.model = v[1].from_pretrained(self.hparams.encoder, output_past=True)

        if self.encoder_tokenizer is None or self.decoder_tokenizer is None:
            raise ValueError("Invalid encoder / decoder params, allowed values %s" %
                             model_dict.keys())

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        input_ids, label_ids, input_mask, label_mask = batch
        y_ids = label_ids[:, :-1].contiguous()
        lm_labels = label_ids[:, 1:].clone()
        lm_labels[label_ids[:, 1:] == self.encoder_tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=y_ids,
            lm_labels=lm_labels
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_nb):
        loss = self._step(batch)
        log = {'train/loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, _ = batch
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=input_mask,
            num_beams=self.hparams.num_beams,
            min_length=self.model.config.min_length,
            bos_token_id=self.encoder_tokenizer.pad_token_id,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            max_length=self.hparams.max_length,
            repetition_penalty=self.hparams.repetition_penalty,
            decoder_start_token_id=self.model.config.decoder_start_token_id
        )
        loss = self._step(batch)

        return {"val_loss": loss, "preds": generated_ids, "target": label_ids}

    def decode(self, ids):
        return self.decoder_tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        outputs_file = os.path.join(self.hparams.model_dir, self.hparams.name, "outputs.json")
        output = []
        for batch in outputs:
            output.extend(self.decode_batch(batch))

        logger.info(json.dumps(output, indent=4))
        with open(outputs_file, 'w') as f:
            json.dump(output, f, indent=4)

        self.calculate_metrics(output)
        return self.test_end(outputs)
