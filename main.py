import pandas as pd
import os, sys, json, pdb
# from torch.utils.tensorboard import SummaryWriter
from data_processing.data_layer import DataParsing
from data_processing.data_loader import PIIDataset, CustomCollateFn
from utils.misc import flatten_out_list, build_vocab
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from models.bilstm_crf import BiLSTM_CRF
from models.early_stop import EarlyStopping
from process_loops.train_text import train, evaluate, predict
import torch.optim as optim


# pd.set_option('display.max_rows', None)

def load_data(p):
    df = (pd.read_json(p)
          .head(400)
          .reset_index()
          )
    return df


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
    train_path = 'data/train.json'
    test_path = 'data/test.json'
    embedding_dim = 128
    hidden_dim = 256
    pad_token_index = 0

    num_layers = 1
    dropout_rate = 0.001
    num_epochs = 20
    learning_rate = 0.005

    df_train = load_data(train_path)
    df_test = load_data(test_path)

    dp = DataParsing(device=device,
                     batch_size=batch_size,
                     chunk_size=chunk_size,
                     overlap=overlap)

    x_train, x_val = train_test_split(df_train,
                                      test_size=test_size,
                                      random_state=random_state
                                      )

    x_train = dp.fit_transform(x_train)
    x_val = dp.fit_transform(x_val)
    x_test = dp.fit_transform(df_test)

    vocab_tokens = build_vocab(x_train.tokens.to_list())
    labels_tokens = build_vocab(x_train.labels.to_list())
    pos_tokens = build_vocab(x_train.pos.to_list())

    vocab_size = len([i for i in vocab_tokens.keys()])
    pos_vocab_size = len([i for i in pos_tokens.keys()])
    num_tags = len([i for i in labels_tokens.keys()])

    model = BiLSTM_CRF(vocab_size=vocab_size,
                       pos_vocab_size=pos_vocab_size,
                       embedding_dim=embedding_dim,
                       hidden_dim=hidden_dim,
                       num_tags=num_tags,
                       num_layers=num_layers,
                       pad_token_index=pad_token_index,
                       dropout_rate=dropout_rate,
                       ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    x_train_t = PIIDataset(x_train, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens, label_to_idx=labels_tokens)
    x_val_t = PIIDataset(x_val, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens, label_to_idx=labels_tokens)
    x_test_t = PIIDataset(x_test, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens)

    collate_fn = CustomCollateFn(chunk_size=chunk_size, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens,
                                 label_to_idx=labels_tokens)
    train_loader = DataLoader(x_train_t, batch_size=batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(x_val_t, batch_size=batch_size, collate_fn=collate_fn)
    predict_loader = DataLoader(x_test_t, batch_size=batch_size, collate_fn=collate_fn)

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Predictions
    # load model crude implementations

    checkpoint = torch.load("checkpoint.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for tokens, pos_tags in predict_loader:
            tokens, pos_tags = tokens.to(device), pos_tags.to(device)
            predictions.append(model(tokens, pos_tags))


    pdb.set_trace()


