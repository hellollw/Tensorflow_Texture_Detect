# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-08

"""
固定格式的csv文件读取操作

"""
import csv
import numpy as np


# 读取固定的csv样本数据（创建文件名列表）
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

# 使用csv写入文件
# 输入：文件名:dataname, 列表数据:datalist
# 输出：在指定位置处写入指定姓名的文件
def csvWrite(dataname, datalist):
    f = open(dataname, 'w', encoding='utf-8', newline='')  # 设置newline=''，不会产生空行
    csv_writer = csv.writer(f)
    for cur_data in datalist:  # datalist应为二维数组
        csv_writer.writerow(cur_data)
    f.close()
    print('写出' + dataname + '成功')