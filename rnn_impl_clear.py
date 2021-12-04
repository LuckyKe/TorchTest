#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256
# 输入形状为(时间步数, 批量大小, 输入个数)
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
# nn.RNN在前向计算后会分别返回输出和隐藏状态h，其中输出指的是隐藏层在各个时间步
# 上计算并输出的隐藏状态，它们通常作为后续输出层的输入。需要强调的是，
# 该"输出"本身并不涉及输出层计算，形状为(时间步数,批量大小,隐藏单元个数)
"""
而nn.RNN实例在前向计算返回的隐藏状态指的是隐藏层在最后时间步的隐藏状态
当隐藏层有多层时，每一层的隐藏状态都会记录在该变量中；对于像长短期记忆，
隐藏状态是一个元组(h,c)，即hidden state和cell state。
"""
num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps, batch_size, vocab_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        X = d2l.to_onehot(inputs, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


model = RNNModel(rnn_layer, vocab_size).to(device)
print(d2l.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))


num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)
