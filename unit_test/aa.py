#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/01/27
    Description:
"""
import math

result = []


def sushu(i):
    i1 = int(i / 10)
    i2 = int(i / 100)
    i3 = int(i / 1000)
    if i3 not in [2, 3, 5, 7]:
        return False
    for n in range(2, int(math.sqrt(i))):
        bool1 = (i % n == 0)
        bool2 = ((i1) % n == 0)
        bool3 = ((i2) % n == 0)
        if n == i2:
            bool3 = False

        ## bool1 为true，说明
        if bool1 or bool2 or bool3:
            return False

    return True


print(sushu(2333))
# print(sushu(1009))

for i in range(1000, 9999):
    bool = sushu(i)
    if bool == True:
        result.append(i)

print(result)
