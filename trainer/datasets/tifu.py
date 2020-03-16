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
                if document['tldr'] is None: continue
                data.append(document)
                if self.hparams.max_documents and len(data) >= self.hparams.max_documents:
                    break

        self.inputs = [
            torch.tensor(self.tokenizer.encode(doc['selftext']))[:self.hparams.max_tokens]
            for doc in data
        ]
        self.labels = [
            torch.tensor(self.tokenizer.encode(doc['tldr']))[:self.hparams.max_tokens]
            for doc in data
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def collate(self, batch):
        inputs = [elem[0] for elem in batch]
        labels = [elem[1] for elem in batch]
        if self.hparams.model_type in ['bart']:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
        inputs_mask = (inputs_padded != 0).int()
        labels_mask = (labels_padded != 0).int()
        return [inputs_padded, labels_padded, inputs_mask, labels_mask]
