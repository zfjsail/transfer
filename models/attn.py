import torch
import torch.nn as nn
import torch.nn.functional as F


class MulInteractAttention(nn.Module):

    def __init__(self, hidden_input_size, inter_size):
        super(MulInteractAttention, self).__init__()
        self.hidden_size = hidden_input_size

        self.fc_1 = nn.Linear(hidden_input_size, inter_size, bias=False)
        self.fc_2 = nn.Linear(hidden_input_size, inter_size, bias=False)
        self.fc_out = nn.Linear(inter_size, 1)

    def forward(self, src_hidden, dst_hidden):
        hidden1 = self.fc_1(src_hidden)
        hidden2 = self.fc_2(dst_hidden)
        out = self.fc_out(hidden1 * hidden2)
        return out