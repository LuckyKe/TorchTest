#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PyTorch提供的autograd包能够根据输入和前向传播
# 过程自动构建计算图，并执行反向传播。
# 完成计算后，可以调用.backward()来完成所有梯度计算。
# 此Tensor的梯度将累积到.grad属性中。

# 在y.backward()时，如果y是标量，则不需要为backward()
# 传入任何参数；否则，需要传入一个与y同形的Tensor。
# with torch.no_grad()
# Tensor和Function互相结合就可以构建一个记录有整个
# 计算过程的有向无环图（DAG）
# 每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor
# 的Function，就是说该Tensor是不是通过某些运算得到
# 的，若是，则grad_fn返回一个与这些运算相关的对象，否则是None
import torch

# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(x.grad_fn)
#
# y = x + 2
# print(y)
# print(y.grad_fn)
# print(y.requires_grad)
#
# print(x.is_leaf, y.is_leaf)
#
# z = y * y * 3
# out = z.mean()
# print(z, out)
#
# # a = torch.randn(2, 2)
# # a = ((a * 3) / (a - 1))
# # print(a.requires_grad)
# # a.requires_grad_(True)
# # print(a.requires_grad)
# # b = (a * a).sum()
# # print(b.grad_fn)
#
# # 因为out是一个标量，所以调用backward()时不需要指定求导变量
# out.backward()  # 等价于out.backward(torch.tensor(1.))
# print(x.grad)
#
# # torch.autograd这个包就是用来计算一些雅克比矩阵的
# # 乘积的。
# # grad在反向传播过程中是累加的，这意味着每一次运行
# # 反向传播，梯度都会累加之前的梯度，所以一般在反向传播
# # 之前需把梯度清零
# out2 = x.sum()
# out2.backward()
# print(x.grad)
#
# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print(x.grad)

# 不允许张量对张量求导，只允许标量对张量求导，求导结果
# 是和自变量同形的张量
# x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# y = 2 * x
# z = y.view(2, 2)
#
# v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
# z.backward(v)
# print(x.grad)

# x = torch.tensor(1.0, requires_grad=True)
# y1 = x ** 2
# with torch.no_grad():
#     y2 = x ** 3
# y3 = y1 + y2
#
# print(x.requires_grad)
# print(y1, y1.requires_grad)
# print(y2, y2.requires_grad)
# print(y3, y3.requires_grad)
#
# y3.backward()
# print(x.grad)

x = torch.ones(1, requires_grad=True)

print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100

y.backward()
print(x)
print(x.grad)
