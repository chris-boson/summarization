import os

import pytorch_lightning as pl
import argparse
from trainer.trainer import SummarizationModel as Model


def main(args):
    model = Model(args)
    logger = pl.loggers.TensorBoardLogger(args.model_dir, name=args.name)
    trainer = pl.Trainer(
        gpus=args.gpus,
        logger=logger
    )
    trainer.fit(model)

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--home_dir', default=os.getcwd(), type=str, help="Home directory")
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpus', default=0)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    args.input_dir = os.path.join(args.home_dir, 'datasets')
    args.model_dir = os.path.join(args.home_dir, 'models')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
