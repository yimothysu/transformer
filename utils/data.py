"""
Deserialize data from disk to PyTorch dataset and dataloader.
"""

import math
import os


import torch
from torch.utils.data import Dataset

from utils.tokenizer import Tokenizer, build_tokenizer


class NextTokenDataset(Dataset):
    def __init__(self, tokens: list[int], block_size: int, device: str = "cpu"):
        self.block_size = block_size
        self.tokens = torch.Tensor(tokens).to(dtype=torch.long, device=device)

    def __len__(self):
        return len(self.tokens) - self.block_size + 1

    def __getitem__(self, idx):
        X = self.tokens[idx : idx + self.block_size - 1]
        y = self.tokens[idx + 1 : idx + self.block_size]
        return X, y


def build_datasets_and_tokenizer(
    data_dir: str, block_size: int, train_size: float = 0.9, device: str = "cpu"
) -> tuple[Dataset, Dataset, Tokenizer]:
    """
    train_size is the proportion of the dataset to include in the train split.
    """

    tokens = []
    tokenizer = build_tokenizer(data_dir)
    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), "r") as f:
            text = f.read()
            doc_tokens = tokenizer.encode(text)
            tokens.extend(doc_tokens)

    train_threshold_index = math.floor(len(tokens) * train_size)
    train_tokens = tokens[:train_threshold_index]
    test_tokens = tokens[train_threshold_index:]
    train_ds = NextTokenDataset(train_tokens, block_size, device=device)
    test_ds = NextTokenDataset(test_tokens, block_size, device=device)

    return train_ds, test_ds, tokenizer
