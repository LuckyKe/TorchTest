#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)

# Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加
# 到计算图中
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)

net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))


net = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
]))

print(net)
print(net[0])

for param in net.parameters():
    print(param)

# torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本
# 可使用input.unsqueeze(0)来添加一维
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# 均方误差损失作为模型的损失函数
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# optimizer = optim.SGD([
#     # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#     {'params': net.subnet1.parameters()},
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
# ], lr=0.03)
#
# # 动态调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零
        l.backward()
        optimizer.step()

    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
