import os

import pytorch_lightning as pl
import argparse
from trainer.models.base import SummarizationModel
from trainer.models.gpt2_lm import GPT2LMSummarizer
from trainer.models.encoder_decoder import EncoderDecoderSummarizer
from trainer.models.bart import BartSummarizer

MODEL_DICT = {
    'gpt2': GPT2LMSummarizer,
    'encoder_decoder': EncoderDecoderSummarizer,
    'bart': BartSummarizer
}

def main(args):
    model = MODEL_DICT[args.model_type](args)
    logger = pl.loggers.TensorBoardLogger(args.model_dir, name=args.name)
    trainer = pl.Trainer(
        amp_level='O1',
        precision=args.precision,
        gpus=args.gpus,
        logger=logger,
        max_epochs=args.max_epochs
    )
    trainer.fit(model)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModel.add_args(parser)
    parser.add_argument('--home_dir', default=os.getcwd(), type=str, help="Home directory")
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--test_percentage', default=0.1, type=float, help="Test percentage.")
    parser.add_argument('--model_type', default='gpt2', type=str,
        choices=['gpt2', 'encoder_decoder', 'bart'], help="Model type.")
    args = parser.parse_args()
    args.input_dir = os.path.join(args.home_dir, 'datasets')
    args.model_dir = os.path.join(args.home_dir, 'models')
    args.max_epochs = int(args.max_epochs)
    args.precision = int(args.precision)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
