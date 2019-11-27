#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : modules.py
# Author            : Zhepei Wang <zhepeiw03@gmail.com>
# Date              : 27.11.2019
# Last Modified Date: 27.11.2019
# Last Modified By  : Zhepei Wang <zhepeiw03@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

class LSTM_Net(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTM_Net, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1,
                           bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths=None):
        x = self.emb(x)
        if lengths is not None:
            x_pack = pack_padded_sequence(x, lengths, batch_first=True)
            tmp, (hn, cn) = self.lstm(x_pack)
            tmp, _ = pad_packed_sequence(tmp, batch_first=True)
        else:
            tmp, (hn, cn) = self.lstm(x)
        out = torch.sigmoid(self.fc(hn[-1]))
        return out

