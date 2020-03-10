import os

import pytorch_lightning as pl
import argparse
from trainer.trainer import SummarizationModel as Model


def main(args):
    model = Model(args)
    logger = pl.loggers.TensorBoardLogger(args.model_dir, name="summarization")
    trainer = pl.Trainer(
        gpus=args.n_gpus,
        logger=logger
    )
    trainer.fit(model)

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--n_gpus', default=1)
    parser.add_argument('--home_dir', default=os.getcwd(), type=str, help="Home directory")
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    args.input_dir = os.path.join(args.home_dir, 'datasets')
    args.model_dir = os.path.join(args.home_dir, 'models')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
