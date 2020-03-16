# Taken from here:
# https://github.com/acarrera94/transformers/blob/bart_summarization_finetuning/examples/summarization/bart/utils.py
# To download the dataset follow instructions here:
# https://github.com/acarrera94/transformers/tree/bart_summarization_finetuning/examples/summarization/bart

import os

from torch.utils.data import Dataset

class CnnDailyMailDataset(Dataset):
    def __init__(self, hparams, tokenizer, type_path="train", block_size=1024):
        super().__init__()
        self.hparams = hparams
        data_dir = os.path.join(self.hparams.input_dir, 'cnn-dailymail/cnn_dm')
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        print("loading " + type_path + " source.")

        with open(os.path.join(data_dir, type_path + ".source"), "r") as f:
            for text in f.readlines():  # each text is a line and a full story
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=block_size, pad_to_max_length=True, return_tensors="pt"
                )
                self.source.append(tokenized)

        print("loading " + type_path + " target.")

        with open(os.path.join(data_dir, type_path + ".target"), "r") as f:
            for text in f.readlines():  # each text is a line and a summary
                tokenized = tokenizer.batch_encode_plus(
                    [text], max_length=56, pad_to_max_length=True, return_tensors="pt"
                )
                self.target.append(tokenized)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()

        src_mask = self.source[index]["attention_mask"].squeeze()  # might need to squeeze

        return source_ids, target_ids, src_mask, None