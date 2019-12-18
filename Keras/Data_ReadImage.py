# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-11

"""
使用tensorflow读取图片文件

需要自己设定裁剪的图片的宽度和高度

步骤：
1. 建立样本文件名列表
2. 建立样本标签列表(可以为one_hot格式)
3. 制造数据集
4. 打包成数据集元组（维度0上所对应)
5. 初始化batch的各个参数

改正：
    1.沿中心裁剪
    2.构建训练集,验证集和测试集（比率为0.6:0.2:0.2)——将训练集，验证集和测试集固定
    3.测试集不能使用batch来进行，直接输入对应的数组矩阵和标签矩阵(在程序内部运行一个sess）
    4.加入一定的数据增强方式（数据增强效果表现不好）
"""

import tensorflow as tf
import random
import pathlib
import numpy as np
import CsvFunction as CSV
import csv


# 图片处理函数
# 尝试数据增强，效果不好
def preProcessImage(imgpath, image_height=224, image_weight=224):
    raw_image = tf.io.read_file(imgpath)
    image_tensor = tf.image.decode_image(raw_image)
    image_process = tf.image.random_flip_up_down(image_tensor) #随机翻转
    image_process = tf.image.random_brightness(image_process,max_delta=30)  #随机亮度
    # image_process = tf.image.random_contrast(image_process,lower=0.2,upper=1.8) #随机对比度
    image_process = tf.image.random_crop(image_process,[image_height,image_weight,3])    #随机裁剪
    # image_process = tf.image.per_image_standardization(image_process)   #标准化
    # image_process = tf.image.resize_image_with_crop_or_pad(image_tensor, image_height, image_weight)  # 以中心为标准裁剪
    image_process = image_process / 255  # 将Image归一化，至[0,1]之间
    return image_process

def preProcessImage_Valid(imgpath,image_height=224, image_weight=224):
    raw_image = tf.io.read_file(imgpath)
    image_tensor = tf.image.decode_image(raw_image)
    image_process = tf.image.resize_image_with_crop_or_pad(image_tensor, image_height, image_weight)  # 以中心为标准裁剪
    # image_process = tf.image.per_image_standardization(image_process)
    image_process = image_process / 255  # 将Image归一化，至[0,1]之间
    return image_process

# 转换int型
def str2int(str_list):
    int_list = []
    for cur_str in str_list:
        int_list.append(int(cur_str))
    return int_list

# 转换为one_hot标签
def getOne_hot(index_list, class_num=47):  # 传入了一个张量
    m = len(index_list)
    one_hot = np.zeros((m, class_num))
    for row in range(m):
        one_hot[row, int(index_list[row])] = 1  #写入csv文件后默认读取为字符串型
    return one_hot


# 通过路径得到标签
def getLabelFromPath(dirpath, all_image_paths):
    dirpath = pathlib.Path(dirpath)
    label_names = [labelnames.name for labelnames in dirpath.glob('*/') if labelnames.is_dir()]
    # class_num = len(label_names)
    # 构造字典索引
    label_dict = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_dict[pathlib.Path(path).parent.name] for path in all_image_paths]
    return all_image_labels


# 读取固定csv文件
def csvRead(usage):
    paths = []
    labels = []
    if usage == 'train':
        f = open('./data_csv/train_path_label.csv', 'r')
    elif usage == 'valid':
        f = open('./data_csv/valid_path_label.csv', 'r')
    elif usage == 'test':
        f = open('./data_csv/test_path_label.csv', 'r')
    else:
        raise NameError('输入错误')
    csv_reader = csv.reader(f)
    for line in csv_reader:
        paths.append(line[0])
        labels.append(line[1])
    return paths, labels


# 构造固定的训练集，验证集和测试集
def getFixDataSet(dirpath, valid_ratio=0.2, test_ratio=0.2):
    dirpath = pathlib.Path(dirpath)
    total_path = list(dirpath.glob('*/*'))  # 全部作为训练样本了，应该作为两个dataset(训练集和测试集)
    total_path = [str(path) for path in total_path]
    total_sample = len(total_path)
    print('样本总数为：' + str(total_sample))
    valid_sample = int(total_sample * valid_ratio)
    test_sample = int(total_sample * test_ratio)
    train_sample = total_sample - test_sample - valid_sample
    print('训练集总数%d 验证集总数%d 测试集总数%d' % (train_sample, valid_sample, test_sample))
    random.shuffle(total_path)  #总样本集和已经打乱
    # 将数据集和拆分，分为训练集(train),验证集(valid)和测试集(test)
    test_path = total_path[:test_sample]
    valid_path = total_path[test_sample:test_sample + valid_sample]
    train_path = total_path[test_sample + valid_sample:total_sample]
    # 生成图片数据对应的图片标签
    train_labels = getLabelFromPath(dirpath, train_path)
    valid_labels = getLabelFromPath(dirpath, valid_path)
    test_labels = getLabelFromPath(dirpath, test_path)
    # 将文件合成，用csv写入(利用zip合成)
    train_path_label = zip(train_path, train_labels)
    valid_path_label = zip(valid_path, valid_labels)
    test_path_label = zip(test_path, test_labels)
    # 文件写入
    CSV.csvWrite('./data_csv/train_path_label.csv', train_path_label)
    CSV.csvWrite('./data_csv/valid_path_label.csv', valid_path_label)
    CSV.csvWrite('./data_csv/test_path_label.csv', test_path_label)
    print('训练集，测试集，验证集的路径，标签写入完成')
    return train_path, valid_path, test_path, train_labels, valid_labels, test_labels


# 生成训练集和验证集的batch
def getDataSet(dirpath, batch_size, valid_ratio=0.2, test_ratio=0.2):
    # 取出固定的训练集和验证集
    try:
        train_path, train_labels = csvRead('train')
        valid_path, valid_labels = csvRead('valid')
        test_path, test_labels = csvRead('test')
    except:
        train_path, valid_path, test_path, train_labels, valid_labels, test_labels = getFixDataSet(dirpath, valid_ratio,
                                                                                                   test_ratio)
    finally:
        print('文件读取完成')
    # 将训练集核测试集融合
    train_path = train_path+test_path
    train_labels = train_labels+test_labels
    print('训练集个数为:%d'%len(train_path))
    print('验证机个数为:%d'%len(valid_path))
    train_image_labels = getOne_hot(train_labels, 47)
    # 获得样本图片路径的dataset
    train_image_paths = tf.data.Dataset.from_tensor_slices(train_path)
    # 获得样本图片标签的dataset
    all_image_labels = tf.data.Dataset.from_tensor_slices(train_image_labels)
    # 预处理
    image_ds = train_image_paths.map(preProcessImage_Valid)   #只有在网络训练的时候用数据增强,增强效果不好，都改为原先模式
    # 结合成一个dataset,使0维度对应
    image_label_ds = tf.data.Dataset.zip((image_ds, all_image_labels))
    ds = image_label_ds.shuffle(buffer_size=1024)
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size)
    train_ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)  # 从随即缓冲区中拿取元素

    # 生成测试集batch(两个batch完全一致）
    valid_sample = len(valid_path)
    valid_image_label = getOne_hot(valid_labels, 47)
    valid_image_paths = tf.data.Dataset.from_tensor_slices(valid_path)
    valid_image_labels = tf.data.Dataset.from_tensor_slices(valid_image_label)
    image_valid = valid_image_paths.map(preProcessImage_Valid)
    image_label_valid = tf.data.Dataset.zip((image_valid, valid_image_labels))
    ds_valid = image_label_valid.shuffle(buffer_size=1024)
    ds_valid = ds_valid.repeat()
    ds_valid = ds_valid.batch(batch_size=int(valid_sample/6))
    ds_valid = ds_valid.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    print('训练集，验证集构造完成')
    return train_ds, ds_valid

# 获得kmeans聚类的数据集（Kmeans聚类数据为4512个）
def getKmeansDataSet(batch_size):
    """
    不需要打乱生成顺序
    :param batch_size:每次读取的样本数量
    :return: Kmeans训练时需要的batch（将之前训练集和验证集结合在一起）
    """
    train_path, train_labels = csvRead('train')
    valid_path, valid_labels = csvRead('valid')
    # 将训练集和验证集和组合在一起
    Kmeans_path = train_path+valid_path
    Kmeans_label = train_labels+valid_labels
    # 组合成batch
    Kmeans_sample = len(Kmeans_path)
    print('Kmeans训练集个数为:%d'%Kmeans_sample)
    Kmeans_label = str2int(Kmeans_label)    #svm分类不需要onehot标签,但需要int型变量
    Kmeans_label = tf.data.Dataset.from_tensor_slices(Kmeans_label)
    Kmeans_path = tf.data.Dataset.from_tensor_slices(Kmeans_path)
    Kmeans_image = Kmeans_path.map(preProcessImage_Valid)
    image_label_Kmeans = tf.data.Dataset.zip((Kmeans_image,Kmeans_label))
    Kmeans_ds = image_label_Kmeans.repeat()
    Kmeans_ds = Kmeans_ds.batch(batch_size=batch_size)
    Kmeans_ds = Kmeans_ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return Kmeans_ds

# 生成测试集数据(cnn or bow)
def getTestDateSet(kind):
    test_path, test_image_label = csvRead('test')
    test_sample = len(test_image_label)
    print('测试集数量为%d'%test_sample)
    if kind == 'cnn':   #cnn需要进行onehot类型转换
        test_image_label = getOne_hot(test_image_label, 47)
    elif kind == 'bow':
        test_image_label = str2int(test_image_label)    #bow需要进行整形转换
    test_image_path = tf.data.Dataset.from_tensor_slices(test_path)
    test_image_label = tf.data.Dataset.from_tensor_slices(test_image_label)
    test_image = test_image_path.map(preProcessImage_Valid)
    image_label_test = tf.data.Dataset.zip((test_image, test_image_label))
    ds_test = image_label_test.repeat(1)
    ds_test = ds_test.batch(batch_size=test_sample)
    iterator = ds_test.make_initializable_iterator()
    data_element = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    test_image, test_labels = sess.run(data_element)
    sess.close()
    print('测试集数据处理完成')
    return test_image, test_labels

if __name__ == '__main__':
    dirpath = 'C:/Users/1/Desktop/LBP/temp'
    batch_size = 5
    buffersize = 32
    epoches = 4
    step_per_epoch = 3
    train_ds,valid_ds = getDataSet(dirpath,batch_size)
    # testimage,testlabels = getTestDateSet()
    # print(np.shape(testimage))
    # print(np.shape(testlabels))
    # ds, test_image, test_label = getDataSet(dirpath, batch_size)
    # iterator = ds.make_initializable_iterator()
    # data_element = iterator.get_next()
    #
    # sess = tf.Session()
    # sess.run(iterator.initializer)
    # for i in range(epoches):
    #     for j in range(step_per_epoch):
    #         time_end = time.time()
    #         print('use:%.8s s'%(time_end-time_start))
    #         print('epoch:%d,step:%d' % (i, j))
    #         image,label = sess.run(data_element)
    #         print(np.shape(image))
    # sess.close()
