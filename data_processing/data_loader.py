from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch

import torch
from torch.utils.data import Dataset


class PIIDataset(Dataset):
    """
    TODO
    dataset for working with data
    """
    def __init__(self, data, word_to_idx, pos_to_idx, label_to_idx=None, device = 'cpu'):
        self.data = data
        self.word_to_idx = word_to_idx
        self.pos_to_idx = pos_to_idx
        self.label_to_idx = label_to_idx
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        tokens_tensor = torch.tensor(
            [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in record["tokens"]], dtype=torch.long, device=self.device)
        pos_tags_tensor = torch.tensor(
            [self.pos_to_idx.get(pos, self.pos_to_idx["<UNK>"]) for pos in record["pos"]], dtype=torch.long, device=self.device)
        if self.label_to_idx is not None:
            labels_tensor = torch.tensor(
                [self.label_to_idx.get(label, self.label_to_idx["<UNK>"]) for label in record["labels"]],
                dtype=torch.long, device=self.device)
            record_tensors = tokens_tensor, pos_tags_tensor, labels_tensor
        else:
            record_tensors = tokens_tensor, pos_tags_tensor
        return record_tensors


class CustomCollateFn:
    """
    TODO
    custom loader with the option to set the key for padding
    """
    def __init__(self, chunk_size, word_to_idx, pos_to_idx, label_to_idx=None):
        self.word_pad = word_to_idx["<PAD>"]
        self.pos_pad = pos_to_idx["<PAD>"]
        self.label_to_idx = label_to_idx
        self.chunk_size = chunk_size

    def _create_tensor(self, pad, batch):
        return torch.stack(
            [torch.cat([t, torch.full((max(0, self.chunk_size - t.size(0)),), pad, dtype=t.dtype)])[
             :self.chunk_size]
             for t in batch])

    def __call__(self, batch):
        tokens_batch, pos_tags_batch = zip(*[(i[0], i[1]) for i in batch])

        tokens_padded = self._create_tensor(self.word_pad, tokens_batch)
        pos_tags_padded = self._create_tensor(self.pos_pad, pos_tags_batch)

        if self.label_to_idx is not None:
            labels_batch = [i[2] for i in batch]
            label_pad = self.label_to_idx["<PAD>"]

            labels_padded = self._create_tensor(label_pad, labels_batch )
            results = tokens_padded, pos_tags_padded, labels_padded
        else:
            results = tokens_padded, pos_tags_padded, None
        return results
