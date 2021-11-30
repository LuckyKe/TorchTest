#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
# x = torch.empty(5, 3)
# print(x)

# x = torch.rand(5, 3)
# print(x)

# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# x = torch.tensor([5.5, 3])
# print(x)
#
# x = x.new_ones(5, 3, dtype=torch.float64)
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)
# print(x)
#
# print(x.size())
# print(x.shape)

# x = torch.arange(1, 8, 2)
# print(x)
#
# x = torch.linspace(1, 8, 2)
# print(x)

# x = torch.randn(1)
# print(x)
# print(x.item())

# x = torch.arange(1, 3).view(1, 2)
# print(x)
# y = torch.arange(1, 4).view(3, 1)
# print(y)
# print(x + y)

# 运算的内存开销
# 索引操作是不会开辟新内存的，而像y=x+y这样的运算是会新开内存的
# 将y指向新内存。
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y = y + x
# print(id(y) == id_before)

# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y[:] = y + x
# print(id(y) == id_before)

# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# torch.add(x, y, out=y)
# print(id(y) == id_before)

# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# id_before = id(y)
# y.add_(x)
# print(id(y) == id_before)

# 虽然view返回的Tensor与源Tensor是共享data的，但是依然是
# 一个新的Tensor（因为Tensor除了包含data外还有一些其他属性）

# x = torch.tensor([1, 2])
# y = x.view(2, 1)
# print(id(x) == id(y))

# Tensor和Numpy互相转换
a = torch.ones(5)
b = a.numpy()
print(a, b)
print(id(a) == id(b))

a += 1
print(a, b)
b += 1
print(a, b)
