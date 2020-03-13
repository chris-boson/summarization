import argparse
import json
import os

import torch
from torch.utils.data import Dataset

from transformers import GPT2Tokenizer


class TIFUDataset(Dataset):
    def __init__(self, hparams: argparse.Namespace):
        # TODO: Make train_test functional
        self.hparams = hparams
        input_path = os.path.join(
            self.hparams.input_dir, 'tifu_all_tokenized_and_filtered.json'
        )
        self.data = []
        with open(input_path) as f:
            for idx, line in enumerate(f):
                document = json.loads(line)
                if document['tldr'] is None: continue
                self.data.append(document)
                if self.hparams.max_documents and idx > self.hparams.max_documents: break
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    @staticmethod
    def add_args(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_tokens', default=1024, type=int, help="Maximum number of input tokens.")
        parser.add_argument('--max_documents', default=None, type=int, help="Maximum number of documents.")
        return parser

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = torch.tensor(self.tokenizer.encode(self.data[index]['selftext']))
        input = input[:self.hparams.max_tokens]
        label = torch.tensor(self.tokenizer.encode(self.data[index]['tldr']))
        return input, label

    @staticmethod
    def collate(batch):
        inputs = [elem[0] for elem in batch]
        labels = [elem[1] for elem in batch]
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        inputs_mask = (inputs_padded != 0).float()
        labels_mask = (labels_padded != 0).float()
        return [inputs_padded, labels_padded, inputs_mask, labels_mask]
