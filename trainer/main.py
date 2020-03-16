import os

import pytorch_lightning as pl
import argparse
from trainer.models.base import SummarizationModel
from trainer.models.gpt2_lm import GPT2LMSummarizer
from trainer.models.encoder_decoder import EncoderDecoderSummarizer
from trainer.models.bart import BartSummarizer

from trainer.datasets.tifu import TIFUDataset
from trainer.datasets.cnn_dm import CnnDailyMailDataset

MODEL_DICT = {
    'gpt2': GPT2LMSummarizer,
    'encoder_decoder': EncoderDecoderSummarizer,
    'bart': BartSummarizer
}

DATASET_DICT = {
    'tifu': TIFUDataset,
    'cnn_dm': CnnDailyMailDataset
}

def main(args):
    dataset_class = DATASET_DICT[args.dataset]
    model = MODEL_DICT[args.model_type](args, dataset_class=dataset_class)
    logger = pl.loggers.TensorBoardLogger(args.model_dir, name=args.name)
    trainer = pl.Trainer(
        amp_level='O1',
        precision=args.precision,
        gpus=args.gpus,
        logger=logger,
        max_epochs=args.max_epochs,
        distributed_backend=args.distributed_backend
    )
    trainer.fit(model)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModel.add_args(parser)
    parser.add_argument('--home_dir', default=os.getcwd(), type=str, help="Home directory.")
    parser.add_argument('--dataset', default='tifu', choices=['tifu', 'cnn_dm'], type=str, help="Which dataset to use.")
    parser.add_argument('--name', type=str, required=True, help="Location for logging and model outputs under home_dir.")
    parser.add_argument('--test_percentage', default=0.1, type=float, help="Test percentage.")
    parser.add_argument('--model_type', default='gpt2', type=str,
        choices=['gpt2', 'encoder_decoder', 'bart'], help="Model type.")
    parser.add_argument('--local_rank', type=int, required=False, help="Required for distributed_backend='ddp'.")
    args = parser.parse_args()
    args.input_dir = os.path.join(args.home_dir, 'datasets')
    args.model_dir = os.path.join(args.home_dir, 'models')
    args.max_epochs = int(args.max_epochs)
    args.precision = int(args.precision)
    args.accumulate_grad_batches = int(args.accumulate_grad_batches)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
