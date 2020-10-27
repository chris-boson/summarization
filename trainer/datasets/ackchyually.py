import argparse
import json
import logging
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from common.aws import download_from_s3

logger = logging.getLogger("lightning")


class AckchyuallyDataset(Dataset):
    def __init__(self, hparams: argparse.Namespace, tokenizer):
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.data_dir = os.path.join(self.hparams.input_dir, "ackchyually")

        self.inputs, self.labels = [], []
        self.load_data()

    def load_data(self):
        self.maybe_download()
        input_path = os.path.join(self.data_dir, "data_2be0547_1601535600_1603180800.tsv")
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path, sep="\t")
        df = df.head(self.hparams.max_documents)
        logger.info(df)
        tqdm.pandas()
        encoded_rows = df.progress_apply(self.encode_row, axis=1)
        self.inputs = [row[0] for row in encoded_rows]
        self.labels = [row[1] for row in encoded_rows]

    def maybe_download(self):
        if not os.path.exists(self.data_dir):
            s3_path = f"s3://dynasty-brain/datasets/{self.hparams.dataset}"
            logger.info(
                f"No cached data found in {self.data_dir}, downloading from {s3_path}"
            )
            download_from_s3(
                s3_path=s3_path,
                local_location=self.data_dir,
            )
        else:
            logger.info(f"Found cached data in {self.data_dir}.")


    def encode_row(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        context = row["text"].split("\n")[-1]
        inputs = torch.tensor(self.tokenizer.encode(
                    f"{context}\nLISA: {row['orginalBody']}",
                    add_space_before_punct_symbol=False
                ),
                dtype=torch.long)[:self.hparams.max_tokens]
        labels = torch.tensor(self.tokenizer.encode(
                    f"{context}\nLISA: {row['editedBody']}",
                    add_space_before_punct_symbol=False
                ),
                dtype=torch.long)[:self.hparams.max_tokens]
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
