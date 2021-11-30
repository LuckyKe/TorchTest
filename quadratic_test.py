#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
# print(math.sqrt(2))
#
#
# def quadratic(a, b, c):
#     delta = math.sqrt(b*b-4*a*c)
#     return (-b+delta)/(2*a), (-b-delta)/(2*a)
#
#
# print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
# print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))
#
# if quadratic(2, 3, 1) != (-0.5, -1.0):
#     print('测试失败')
# elif quadratic(1, 3, -4) != (1.0, -4.0):
#     print('测试失败')
# else:
#     print('测试成功')


def add_end(L=None):
    if L is None:
        L = []
    L.append('END')
    return L


print(add_end())
print(add_end())
