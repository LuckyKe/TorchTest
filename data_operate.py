#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# print(np.__version__)
# # np.show_config()
#
# Z = np.zeros(10)
# print(Z)
#
# Z = np.zeros((10, 10))
# print("%d bytes" % (Z.size * Z.itemsize))
#
# Z = np.zeros(10)
# Z[4] = 1
# print(Z)
#
# Z = np.arange(10, 50)
# print(Z)
#
# Z = np.arange(50)
# Z = Z[::-1]
# print(Z)

# Z = np.arange(9).reshape(3, 3)
# print(Z)

# Z = np.eye(3)
# print(Z)

# nz = np.nonzero([1, 2, 0, 0, 4, 0])
# print(nz)

# Z = np.random.random((3, 3, 3))
# print(Z)

# Z = np.random.random((10, 10))
# Zmin, Zmax = Z.min(), Z.max()
# print(Zmin, Zmax)

# Z = np.random.random(30)
# m = Z.mean()
# print(m)

# Z = np.ones((10, 10))
# Z[1:-1, 1:-1] = 0
# print(Z)

# Z = np.ones((5, 5))
# Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
# print(Z)

# print(0 * np.nan)
# print(np.nan == np.nan)
# print(np.inf > np.nan)
# print(np.nan - np.nan)
# print(0.3 == 3 * 0.1)

# Z = np.diag(1 + np.arange(4), k=1)
# print(Z)

# print(np.unravel_index(100, (6, 7, 8)))

# Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
# print(Z)

# Z = np.random.random((5, 5))
# Zmax, Zmin = Z.max(), Z.min()
# Z = (Z - Zmin) / (Zmax - Zmin)
# print(Z)

# color = np.dtype([('r', np.ubyte, 1),
#                   ('g', np.ubyte, 1),
#                   ('b', np.ubyte, 1),
#                   ('a', np.ubyte, 1)])
# print(color)

# Z = np.dot(np.ones((5, 3)), np.ones((3, 2)))
# print(Z)

# Z = np.ones((5, 3))
# print(Z * 2)

# Z = np.arange(11)
# Z[(3 < Z) & (Z <= 8)] *= -1
# print(Z)

# print(sum(range(5), -1))
# from numpy import *
# print(sum(range(5), -1))

Z = np.arange(5)
print(1j*Z)
