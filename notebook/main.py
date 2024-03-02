import pandas as pd
import os, sys, json, pdb
# from torch.utils.tensorboard import SummaryWriter
from data_layer import DataParsing
from data_loader import PIIDataset, CustomCollateFn
from labels_to_data_bilstm_crf import convert_to_labels
from misc import build_vocab, f5_score_mapping
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from bilstm_crf import BiLSTM_CRF
from early_stop import EarlyStopping
from train_text import train, evaluate, predict
import torch.optim as optim
from sklearn.metrics import fbeta_score


# pd.set_option('display.max_rows', None)

def load_data(p):
    df = (pd.read_json(p)
          .loc[:1000, :]
          # .sample(800)
          .reset_index()
          )
    return df


def robin_test(p):
    df = (pd.read_json(p)
          .loc[1000:1500, :]
          .sample(100)
          .reset_index()
          )
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # TODO move to config files,
    # Separate according to process

    chunk_size = 400
    overlap = 2
    test_size = 0.2
    random_state = 42
    batch_size = 16
    train_path = f'{os.path.realpath("..")}/data/train.json'
    test_path = f'{os.path.realpath("..")}data/test.json'
    embedding_dim = 128
    hidden_dim = 256
    pad_token_index = 0
    pre_processing_chunk_size = 5000
    num_layers = 1
    dropout_rate = 0.02
    num_epochs = 20
    learning_rate = 0.003
    patience = 5
    verbose = True
    beta = 5

    df_robin = robin_test(train_path)
    df_train = load_data(train_path).query("~`index`.isin(@df_robin.index)", engine='python').reset_index(drop=True)
    df_robin = df_robin.reset_index(drop=True)
    # todo robin test
    # df_test = load_data(test_path)
    # to remove

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
    # x_test = dp.fit_transform(df_test)

    # todo robin test
    # to remove
    x_robin = dp.fit_transform(df_robin)
    x_robin_y = x_robin.copy()
    x_robin = (x_robin.drop(columns='labels'))

    # pdb.set_trace()
    vocab_tokens = build_vocab(x_train.tokens.to_list())
    labels_tokens = build_vocab(x_train.labels.to_list())
    pos_tokens = build_vocab(x_train.pos.to_list())

    vocab_size = len([i for i in vocab_tokens.keys()])
    pos_vocab_size = len([i for i in pos_tokens.keys()])
    num_tags = len([i for i in labels_tokens.keys()])

    # pdb.set_trace()
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

    # todo robin test
    # x_test_t = PIIDataset(x_test, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens)

    # to remove
    x_robin_t = PIIDataset(x_robin, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens)

    collate_fn = CustomCollateFn(chunk_size=chunk_size, word_to_idx=vocab_tokens, pos_to_idx=pos_tokens,
                                 label_to_idx=labels_tokens)
    train_loader = DataLoader(x_train_t, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    valid_loader = DataLoader(x_val_t, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    # todo robin test
    # predict_loader = DataLoader(x_test_t, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # to remove
    predict_loader = DataLoader(x_robin_t, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

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

    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for tokens, pos_tags in predict_loader:
            tokens, pos_tags = tokens.to(device), pos_tags.to(device)
            predictions.append(model(tokens, pos_tags))

    # todo robin test
    # raw_predictions = convert_to_labels(x_test, predictions, labels_tokens, pre_processing_chunk_size)

    # to remove
    raw_predictions = convert_to_labels(x_robin, predictions, labels_tokens, pre_processing_chunk_size)
    raw_predictions.to_csv('test_pred.csv', index=False)

    scoring = f5_score_mapping(df_y=x_robin_y, df_y_hat=raw_predictions, label_mapping=labels_tokens)

    scoring.to_csv('score_test_pred.csv')
    f5 = fbeta_score(scoring.y_idx.to_list(), scoring.y_hat_idx.to_list(), beta=beta, average='micro')
    print(f'the f5 score for this run: {f5}')
    pdb.set_trace()
