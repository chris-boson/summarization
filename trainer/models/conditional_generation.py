
import json
import os

import torch
from torch.nn import functional as F

from transformers.modeling_bart import BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer
from trainer.models.base import SummarizationModel


class ConditionalGenerationSummarizer(SummarizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.get_tokenizers()
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
            raise ValueError("Invalid encoder / decoder params, allowed values %s" % model_dict.keys())

    def get_tokenizers(self):
        self.get_model()

    def forward(self, input_ids, attention_mask, label_ids, label_mask):
        try:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=label_ids,
                decoder_attention_mask=None#label_mask
            )
        except:
            print("Inputids", input_ids)
            print("attention mask", attention_mask)
            print("label ids", label_ids)
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=label_ids,
                decoder_attention_mask=None#label_mask
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

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_nb):
        input_ids, label_ids, input_mask, _ = batch
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=input_mask,
            # num_beams=3,
            max_length=40,
            repetition_penalty=3.0,
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