import argparse
import json
import os

import torch
from torch.utils.data import Dataset


class TIFUDataset(Dataset):
    def __init__(self, hparams: argparse.Namespace, tokenizer):
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.inputs, self.labels = [], []
        self.load_data()

    def load_data(self):
        input_path = os.path.join(
            self.hparams.input_dir, 'tifu_all_tokenized_and_filtered.json'
        )
        data = []
        with open(input_path) as f:
            for line in f:
                document = json.loads(line)
                if document['tldr'] is None or len(self.tokenizer.encode(document['tldr'])) <= 3:
                    continue
                data.append(document)
                if self.hparams.max_documents and len(data) >= self.hparams.max_documents:
                    break

        self.inputs = [
            torch.tensor(
                self.tokenizer.encode(
                    doc['selftext_without_tldr'],
                    add_space_before_punct_symbol=False
                ),
                dtype=torch.long)[:self.hparams.max_tokens]
            for doc in data
        ]
        self.labels = [
            torch.tensor(
                self.tokenizer.encode(
                    doc['tldr'],
                    add_space_before_punct_symbol=False
                ),
                dtype=torch.long)[:self.hparams.max_tokens]
            for doc in data
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
