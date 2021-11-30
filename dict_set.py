#!/usr/bin/env python
# -*- coding: utf-8 -*-
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
# print(d['Michael'])
# d['Adam'] = 67
# print(d['Adam'])
#
# s = set([1, 1, 2, 2, 3, 3])
# print(s)
# s.add(4)
# print(s)
# s.add(4)
# print(s)
# s.remove(4)
# print(s)

# dict的key必须是不可变对象。这是因为dict根据key来计算value的存储
# 位置，如果每次计算相同的key得出的结果不同，那dict内部就完全混乱了
key = (1, [2, 3])
d[key] = 'a list'
# s1 = set([1, 2, 3])
# s2 = set([2, 3, 4])
# print(s1 | s2)

# a = ['c', 'b', 'a']
# a.sort()
# print(a)
