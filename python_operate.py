#!/usr/bin/env python
# -*- coding: utf-8 -*-
# print(r'''line1
# line2
# line3''')

# a = 1
# t_007 = 'T007'
# Answer = True

# a = 123
# print(a)
# a = 'ABC'
# print(a)
# 变量本身类型不固定，称之为动态语言，与之对应的是静态语言。
# 静态语言在定义变量时必须指定变量类型，如果赋值的时候类型不匹配，就会报错。

# PI = 3.14159265359
# print(10 / 3)
# print(9 / 3)
# print(10 // 3)

# Unicode字符集。Unicode把所有语言都统一到一套编码里，这样就不会
# 有乱码问题了。

# Unicode编码转化成"可变长编码"的UTF-8编码。UTF-8编码把一个Unicode
# 字符根据不同的数字大小编码成1-6个字节，常用的英文字母被编码成1个字节，
# 汉字通常是3个字节，只有很生僻的字符才会被编码成4-6个字节。如果要传输
# 的文本包含大量英文字符，用UTF-8编码就能节省空间

# 现代计算机系统通用的字符编码工作方式：
# 在计算机内存中，统一使用Unicode编码，当需要保存到硬盘或者需要传输的时候
# 就转化为UTF-8编码。
# 用记事本编辑的时候，从文件读取的UTF-8字符被转换为Unicode字符到内存里
# 编辑完成后，保存的时候再把Unicode转换为UTF-8保存到文件

# ord()函数获取字符的整数表示，chr()函数把编码转换为对应的字符
# print('包含中文的str')
# print(ord('A'))
# print(chr(66))
# print(chr(25991))

# print('\u4e2d\u6587')
# 由于Python的字符串类型是str，在内存中以Unicode表示，一个字符对应
# 若干个字节。如果要在网络上传输，或者保存到磁盘上，就需要把str
# 变为以字节为单位的bytes。
# Python对bytes类型的数据用带b前缀的单引号或双引号表示
# print('ABC'.encode('ascii'))
# print('中文'.encode('utf-8'))
# print('中文'.encode('ascii'))

# print(len('ABC'))
# print(len('中文'))
# print(len(b'ABC'))
# print(len('中文'.encode('utf-8')))

# print('Hello, %s' % 'world')
# print('Hi, %s, you have $%d.' % ('Michael', 1000000))

# print('%2d-%02d' % (3, 1))
# print('%.2f' % 3.1415926)

# print('Age: %s. Gender: %s' % (25, True))

# format()方法
# print('Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125))

# f-string
# r = 2.5
# s = 3.14 * r ** 2
# print(f'The area of a circle with radius {r} is {s:.2f}')

# s1 = 72
# s2 = 85
# r = (85 - 72) / 72 * 100
# print('成绩提升了: %.1f%%' % r)


