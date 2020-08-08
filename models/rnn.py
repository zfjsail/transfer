import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size=10000, batch_size=32, embedding_size=128, hidden_size=32, dropout=0.2,
                 multiple=0, pretrain=None):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        # self.msl = max_sequence_length
        self.multiple = multiple
        self.n_d = 192
        self.n_out = 192

        # embedding layer
        self.embed_seq = nn.Embedding(self.vocab_size + 1, embedding_size)
        self.embed_keyword_seq = nn.Embedding(self.vocab_size + 1, embedding_size)
        if pretrain is not None:
            self.embed_seq.weight = nn.Parameter(torch.FloatTensor(pretrain))
            embedding_size = pretrain.shape[1]
        # print(type(self.embed_seq.weight))

        self.embed_seq.weight.requires_grad = False
        self.embed_keyword_seq.weight.requires_grad = False

        # LSTM layer
        self.lstm_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.lstm_seq2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        self.lstm_key_seq1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.lstm_key_seq2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

        # dense layer & normalization
        # self.normalization = nn.BatchNorm1d()
        self.output = nn.Sequential(
            nn.Linear(6 * hidden_size + 3 * multiple, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            # nn.Sigmoid()
        )

    def forward(self, mag, aminer, keyword_mag, keyword_aminer):
        mag = self.embed_seq(mag)
        aminer = self.embed_seq(aminer)
        keyword_mag = self.embed_keyword_seq(keyword_mag)
        keyword_aminer = self.embed_keyword_seq(keyword_aminer)
        # jaccard = jaccard.repeat(1, 2 * self.multiple)
        # inverse = inverse.repeat(1, self.multiple)
        mag, _ = self.lstm_seq1(mag)
        mag, _ = self.lstm_seq2(mag)
        aminer, _ = self.lstm_seq1(aminer)
        aminer, _ = self.lstm_seq2(aminer)
        keyword_mag, _ = self.lstm_key_seq1(keyword_mag)
        keyword_mag, _ = self.lstm_key_seq2(keyword_mag)
        keyword_aminer, _ = self.lstm_key_seq1(keyword_aminer)
        keyword_aminer, _ = self.lstm_key_seq2(keyword_aminer)
        minus = keyword_mag[:, -1, :] - keyword_aminer[:, -1, :]
        minus_key = mag[:, -1, :] - aminer[:, -1, :]
        concat_input = torch.cat(
            (minus,
             minus_key,
             # jaccard,
             # inverse,
             mag[:, -1, :],
             aminer[:, -1, :],
             keyword_mag[:, -1, :],
             keyword_aminer[:, -1, :],
             ), dim=1)

        # print("concat output", concat_input.size(), concat_input)

        output = self.output(concat_input)
        # output = self.sigmoid(output)
        return torch.log_softmax(output, dim=1), concat_input