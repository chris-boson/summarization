import os
import json

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch


class TIFUDataset(Dataset):
    def __init__(self, input_dir):
        input_path = os.path.join(input_dir, 'tifu_all_tokenized_and_filtered.json')
        self.data = []
        with open(input_path) as f:
            for line in f:
                document = json.loads(line)
                if document['tldr'] is None: continue
                self.data.append(document)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = torch.tensor(self.tokenizer.encode(self.data[index]['selftext']))[:1024]
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
