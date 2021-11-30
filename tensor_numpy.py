#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch

# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a, b)
#
# a += 1
# print(a, b)
# b += 1
# print(a, b)
#
# c = torch.tensor(a)
# a += 1
# print(a, c)

# Tensor on GPU
x = torch.rand(5, 3)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))
