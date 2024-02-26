
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, embedding_dim, hidden_dim, num_tags, pad_token_index, num_layers=1,
                 dropout_rate=.001):
        super(BiLSTM_CRF, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.pad_token_index = pad_token_index

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, embedding_dim)

        self.bi_lstm = nn.LSTM(embedding_dim * 2,
                               hidden_dim // 2,
                               num_layers=self.num_layers,
                               bidirectional=True,
                               batch_first=True,
                               dropout=self.dropout_rate if num_layers > 1 else 0)

        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        self.crf = CRF(num_tags, batch_first = True)

    def forward(self, tokens, pos_tags, labels=None):

        token_embeddings = self.token_embedding(tokens)
        pos_embeddings = self.pos_embedding(pos_tags)
        embeddings = torch.cat((token_embeddings, pos_embeddings), dim=-1)

        lstm_out, _ = self.bi_lstm(embeddings)

        emissions = self.hidden2tag(lstm_out)

        if labels is not None:
            return  -self.crf(emissions, labels)
        else:
            return self.crf.decode(emissions)
