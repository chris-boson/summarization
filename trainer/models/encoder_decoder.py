
import os
import json

import torch
from torch.nn import functional as F

from transformers import PreTrainedEncoderDecoder
from transformers import GPT2Tokenizer, BertTokenizer, TransfoXLTokenizer
from trainer.models.base import SummarizationModel


class EncoderDecoderSummarizer(SummarizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.get_model()
        self.get_datasets()

    def get_model(self):
        tokenizer_dict = {
            'gpt2': GPT2Tokenizer,
            'bert': BertTokenizer,
            'transfo-xl': TransfoXLTokenizer
        }
        self.model = PreTrainedEncoderDecoder.from_pretrained(
            self.hparams.encoder, self.hparams.decoder
        )
        for k, v in tokenizer_dict.items():
            if k in self.hparams.encoder:
                self.encoder_tokenizer = v.from_pretrained(self.hparams.encoder)
                if self.encoder_tokenizer.pad_token is None:
                    self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
            if k in self.hparams.decoder:
                self.decoder_tokenizer = v.from_pretrained(self.hparams.decoder)

        if self.encoder_tokenizer is None or self.decoder_tokenizer is None:
            raise ValueError("Invalid encoder / decoder params, allowed values %s" %
                             tokenizer_dict.keys())

    def forward(self, input_ids, attention_mask, label_ids, label_mask):
        return self.model(
            encoder_input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=label_ids,
            decoder_attention_mask=None#label_mask
        )

    def _step(self, batch):
        input_ids, label_ids, input_mask, label_mask = batch

        #TODO: Check if this is the right thing to do
        #Pad label_ids to same length as input_ids
        padding = (torch.zeros(
            (input_ids.shape[0], input_ids.shape[1] - label_ids.shape[1]), dtype=torch.int
        ) + self.encoder_tokenizer.pad_token_id).type_as(label_ids)
        label_ids = torch.cat([label_ids, padding], dim=-1)

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

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, _ = batch
        # TODO: num_beams > 1 doesn't work yet
        generated_ids = self.model.decoder.generate(
            input_ids,
            attention_mask=input_mask,
            num_beams=self.hparams.num_beams,
            max_length=self.hparams.max_length,
            repetition_penalty=self.hparams.repetition_penalty,
            min_length=self.model.decoder.config.min_length,
            bos_token_id=self.decoder_tokenizer.pad_token_id,
            pad_token_id=self.model.decoder.config.pad_token_id,
            eos_token_id=self.model.decoder.config.eos_token_id,
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
            output.extend(
                [{
                    'prediction': self.decode(batch['preds'][i]),
                    'target': self.decode(batch['target'][i])
                }]
                for i in range(len(batch['preds']))
            )
        print(json.dumps(output, indent=4))
        with open(outputs_file, 'w') as f:
            json.dump(output, f, indent=4)

        return self.test_end(outputs)
