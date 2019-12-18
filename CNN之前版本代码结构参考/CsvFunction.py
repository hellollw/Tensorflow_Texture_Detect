# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-08

"""
固定格式的csv文件读取操作

"""
import csv
import numpy as np


# 读取固定的csv样本数据
def csvRead(usage, type):
    """

    :param usage:train or test
    :param type: label or name
    :return:
    """
    data = []
    path = './result/data/' + str(usage) + 'file' + '_' + str(type) + '.csv'
    f = open(path, 'r')
    csv_reader = csv.reader(f)
    for line in csv_reader:
        if type == 'label':
            data.append(line)
        elif type == 'name':
            data.append(line[1])
    return data


def str2float(datastr):
    m, n = np.shape(datastr)
    datafloat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            try:
                datafloat[i][j] = float(datastr[i][j].rstrip())  # 数组遍历实现转换
            except:
                print('为0位置在于%d行%d列' % (i + 1, j + 1))
                datafloat[i][j] = 0.0

    return datafloat
