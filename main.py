import pandas as pd
import os, sys, json, pdb

from data_processing.data_layer import DataParsing
from data_processing.data_loader import PIIDataset, CustomCollateFn
from utils.misc import flatten_out_list, build_vocab
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    chunk_size = 400
    overlap = 2
    test_size = 0.2
    random_state = 42
    batch_size = 16

    dp = DataParsing(chunk_size=chunk_size, overlap=overlap)

    df = (pd.read_json('data/train.json')
          .head(100)
          .reset_index()
          )

    unique_tokens = flatten_out_list(df.tokens.to_list())
    unique_labels = flatten_out_list(df.labels.to_list())

    x_train, x_val = train_test_split(df, test_size=test_size, random_state=random_state)

    x_train = dp.fit_transform(x_train)

    x_val = dp.fit_transform(x_val)

    vocab_tokens = build_vocab(x_train.tokens.to_list())
    labels_tokens = build_vocab(x_train.labels.to_list())
    pos_tokens = build_vocab(x_train.pos.to_list())

    x_train_t = PIIDataset(x_train,
                           word_to_idx=vocab_tokens,
                           pos_to_idx=pos_tokens,
                           label_to_idx=labels_tokens)

    collate_fn = CustomCollateFn(chunk_size=chunk_size,
                                 word_to_idx=vocab_tokens,
                                 pos_to_idx=pos_tokens,
                                 label_to_idx=labels_tokens)
    loader = DataLoader(x_train_t,
                        batch_size= batch_size,
                        collate_fn=collate_fn)

    for t,p,l in loader:
        tokens_padded, pos_tags_padded, labels_padded = t.to(device), p.to(device), l.to(device)
        print("Tokens Shape:", tokens_padded.shape)
        print("POS Tags Shape:", pos_tags_padded.shape)
        print("Labels Shape:", labels_padded.shape)

    # pdb.set_trace()