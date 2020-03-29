
import torch

from transformers import PreTrainedEncoderDecoder
from transformers import GPT2Tokenizer, BertTokenizer, TransfoXLTokenizer
from trainer.models.base import SummarizationModel


class EncoderDecoderSummarizer(SummarizationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
            if k in self.hparams.decoder:
                self.decoder_tokenizer = v.from_pretrained(self.hparams.decoder)

        if self.encoder_tokenizer is None or self.decoder_tokenizer is None:
            raise ValueError("Invalid encoder / decoder params, allowed values %s" %
                             tokenizer_dict.keys())

    def forward(self, input_ids, attention_mask, label_ids, label_mask):
        return self.model(
            encoder_input_ids=input_ids,
            # encoder_attention_mask=attention_mask,
            # decoder_input_ids=input_ids#,
            # decoder_attention_mask=attention_mask
            decoder_input_ids=label_ids,
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