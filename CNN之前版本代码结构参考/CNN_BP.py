# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-30

"""
使用tensorflow实现卷积神经网络编程

步骤：
    1.tensorflow的图像读取(使用之前写好的TF_readImage，图片数据以三通道矩阵读入，标签值以one-hot形式读入）
    2.CNN计算图的搭建：
        2.1 定义占位符（程序的输入值，最终输出值）
        2.2 转变输入数据结构
        2.3 定义卷积层1（权重:weight,偏移量:bias其维度=输出维度，激活函数）（使用卷积运算）
        2.4 定义池化层1
        .
        .（省略中间层数）
        .
        2.5 定义池化层n
        2.6 修改数据格式，全连接层的输入应为向量形式
        2.7 定义全连接层1（神经元个数，权重:weight,偏移量:bias其维度=输出维度，激活函数)（使用矩阵相乘运算）
        2.8 防止过拟合采取tf.nn.dropout方法
        .
        .（省略中间层数）
        .
        2.9 定义全连接输出层（输出向量维度与风雷目标数目相同)
        2.10 定义损失函数
        2.11 定义优化器
        2.12 求取准确率（按照输出向量与标签向量的相似性）
        2.14 初始化变量（需要初始化变量才不会报错）
        2.15 初始化读取数据线程
        2.16 计算图
        2.17 保存图


重点：
    1.每一步数据的维度需要精准的计算好
    2.数据的读取先用计算图取出数据，数据由tensor类型转换为ndarray型，再放入feed中
    3.placeholder占位符定义的维度需要与输入数据的维度一致
    4.在TF中的全局变量和局部变量有着本质区别，定义局部变量需要声明所在的变量集合

问题：
    1. padding='SAME':输出大小等于输入大小除以步长向上取整，s是步长大小.故输出取决于步长大小！！卷积层和池化层的输出（降维关系只和步长有关）！！

"""

import tensorflow as tf
import numpy as np
import TF_readImage as TF_read
import time
import csv


# cnn各层参数
class CNN:
    def __init__(self, num,convlayers, nnlayers, conv_patch, pool_patch, conv1, outchannal1, pool1, conv2, outchannal2,
                pool2,nn1):
        self.num  = num
        self.convlayers = convlayers
        self.nnlayers = nnlayers
        self.conv_patch = conv_patch
        self.pool_patch = pool_patch
        self.conv1 = conv1
        self.outchannal1 = outchannal1
        self.pool1 = pool1
        self.conv2 = conv2
        self.outchannal2 = outchannal2
        self.pool2 = pool2
        self.nn1 = nn1


# 读取固定的csv样本数据
def csvRead(usage, type):
    """

    :param usage:train or test
    :param type: label or name
    :return:
    """
    data = []
    path = './result/data/' + str(usage) +'file'+'_' + str(type) + '.csv'
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


# 产生随机权重变量w
# 输入：想要的变量维度:shape
# 输出：对应维度的变量
def WeightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 产生标准差为0.1的正态分布张量
    return tf.Variable(initial)


# 产生随机偏移量b
# 输入：变量维度:shape
# 输出：对应维度的变量
def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)  # TF的变量赋初值的方式
    return tf.Variable(initial)


# 卷积方式（这里展示默认步长都为1)
def conv2d(x, W, patch):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.conv2d(x, W, strides=[1, patch, patch, 1], padding='SAME')  # 卷积函数，stride为卷积步长，same为考虑边界


# 池化方式：
#   选择最大值池化
#   池化大小选择为：2x2
#   步长选择为:2x2
def max_pool_2x2(x, pool, patch):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.max_pool(x, ksize=[1, pool, pool, 1], strides=[1, patch, patch, 1], padding='SAME')  #
    # ksize为池化窗口大小，strides为池化窗口步长


if __name__ == '__main__':
    start_time = time.time()
    # 读入数据
    # 定义需要使用到的系数
    # path = './temp/'
    # ratio = 0.2
    batchsize = 32
    image_height = 163
    image_weight = 163
    epochs = 6  # 数据增强
    i = 1
    # 定义CNN系数
    cnn = CNN(num=1,convlayers=3, nnlayers=2, conv_patch=1, pool_patch=3, conv1=5, outchannal1=32, pool1=3, conv2=3,
              outchannal2=64, pool2=3,nn1=1028)
    # 获得固定训练数据
    trainfile_name = csvRead('train', 'name')
    trainfile_label_str = csvRead('train', 'label')
    trainfile_label_float = str2float(trainfile_label_str)
    trainlabel_num = np.shape(trainfile_label_str)[1]

    testfile_name = csvRead('test', 'name')
    testfile_label_str = csvRead('test', 'label')
    testfile_label_float = str2float(testfile_label_str)
    print('读取数据完成')
    # 获得随机训练数据
    # trainfile_name, trainfile_label_onehot, trainlabel_num, testfile_name, testfile_label_onehot, testlabel_num = \
    #     TF_read.getTrainAndTestData(path, ratio=0.2)
    # 构造batch
    # image_batch, label_batch = TF_read.getBatch(trainfile_name, trainfile_label_float, batchsize, image_height,
    #                                             image_weight, num_epochs=epochs)  # 训练过程
    image_batch, label_batch = TF_read.getBatch(trainfile_name, trainfile_label_float, batchsize, image_height,
                                                image_weight, num_epochs=epochs)  # 测试过程
    # 构造图
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))  #
    # 先构建session再构造图

    # 预定义输入x，输出y（即预定义占位符——先预定义过程，之后在执行的时候在具体赋值，预定义维度）
    x = tf.placeholder(tf.float32, shape=[None, image_height, image_weight, 1])  # None表示占时不能确定的维度，根据输入的数据改变！
    y = tf.placeholder(tf.float32, shape=[None, trainlabel_num])  # 样本类别数量
    keep_prob = tf.placeholder(tf.float32)  # 减少计算量dropout（防止过拟合，神经元会被以一定概率选中并在这次迭代中不更新权值），只有当keep_prob = 1时，才是所有的神经元都参与工作
    # 调整输入样本结构
    x_image = tf.reshape(x, [-1, image_height, image_weight, 1])  # 输入图片格式：[ batch, in_height, in_weight, in_channel ]

    # 定义卷积层1:
    # 卷积核1：patch = 5x5; in size:1; out size:32; 激活函数:Relu
    W_conv1 = WeightVariable([cnn.conv1, cnn.conv1, 1,
                              cnn.outchannal1])  # 卷积核格式：shape为 [ filter_height, filter_weight, in_channel, out_channels ]
    b_conv1 = biasVariable([cnn.outchannal1])  # 偏移量对应out size
    # print(np.shape(x_image))
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, cnn.conv_patch) + b_conv1)
    # print(np.shape(h_conv1))
    # 定义池化层1
    h_pool1 = max_pool_2x2(h_conv1, cnn.pool1, cnn.pool_patch)  # 输出为14*14*32
    print(np.shape(h_pool1))

    # 定义卷积层2：
    # 卷积核2：patch = 5x5; in size:32; out size:64; 激活函数：Relu
    W_conv2 = WeightVariable([cnn.conv2, cnn.conv2, cnn.outchannal1, cnn.outchannal2])
    b_conv2 = biasVariable([cnn.outchannal2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, cnn.conv_patch) + b_conv2)
    # 定义池化层2
    h_pool2 = max_pool_2x2(h_conv2, cnn.pool2, cnn.pool_patch)  # 输出为7*7*64
    print(np.shape(h_pool2))

    # 卷积层3
    W_conv3 = WeightVariable([5,5,cnn.outchannal2,128])
    b_conv3 = biasVariable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, cnn.conv_patch) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3, cnn.pool2, cnn.pool_patch)
    print(np.shape(h_pool3))

    # 定义全连接层1（1024个神经元）
    weight = int(np.shape(h_pool3)[2])  #只与步长有关
    print(weight*weight*128)
    h_pool3_flat = tf.reshape(h_pool3, [-1, weight * weight * 128])  # 每个样本的特征数量为7*7*64，全连接层应为输入向量
    W_fc1 = WeightVariable([weight * weight * 128, cnn.nn1])  # 特征维度（上一层池化输出）*1024个神经元（设定的神经元个数）
    b_fc1 = biasVariable([cnn.nn1])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # 转换为矩阵相乘
    h_fc1_drop = tf.nn.dropout(h_fc1,
                               keep_prob)  # 减少计算量dropout（防止过拟合，神经元会被以一定概率选中并在这次迭代中不更新权值），只有当keep_prob = 1时，才是所有的神经元都参与工作

    # 定义全连接层2（10个神经元，对应输出维度）
    W_fc2 = WeightVariable([cnn.nn1, trainlabel_num])
    b_fc2 = biasVariable([trainlabel_num])
    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # 结果对应预测结果

    # 定义损失函数（可以直接用向量相减运算）
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))  # 二次代价函数:预测值与真实值的误差,同时求取均值

    # 梯度下降法求解，选用AdamOptimizer优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    print('构造图完成')

    # 求取准确率
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 当前正确率的均值

    # 保存数据
    saver = tf.train.Saver()  # 默认保存所有变量

    # 初始化变量
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # 初始化读取数据线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)  # 把张量tensor推入内存之中
    print('进入训练')
    """
    训练代码
    """
    try:
        i = 1
        while not coord.should_stop():
            image, label = sess.run([image_batch, label_batch])  # 读取数据
            if i % 5 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x: image, y: label, keep_prob: 1.0})  # feed不能为张量
                print('step:%d ' % i + "training accuracy：%f" % train_accuracy)
            train_step.run(feed_dict={x: image, y: label, keep_prob: 1.0})
            i += 1
    except tf.errors.OutOfRangeError:
        print('complete')
    finally:
        coord.request_stop()  # 停止读入线程
    coord.join(threads)  # 线程加入主线程，等待主线程结束
    # saver.save(sess, './model.ckpt')
    end_time = time.time()
    f = open('./result/CNN/CNNrecord.txt','a+')
    f.write('第%d个训练器\n'%cnn.num)
    f.write('训练器参数：卷积层数量%d 全连接层数量%d 卷积层步长%d 池化层步长%d 卷积层1大小%d 卷积层1输出通道%d 池化层1大小%d 卷积层2大小%d 卷积层2输出通道%d 池化层2大小%d '
            '全连接层1大小%d \n'%(cnn.convlayers,cnn.nnlayers,cnn.conv_patch,cnn.pool_patch,cnn.conv1,cnn.outchannal1,
                         cnn.pool1,cnn.conv2,cnn.outchannal2,cnn.pool2,cnn.nn1))
    f.write('程序运行耗时:%.8s'%(end_time-start_time))
    f.close()
    sess.close()
    print('第%d个程序运行耗时为：%.8s s' % (cnn.num,end_time - start_time))
    #
    # """
    # 测试代码
    # """
    # saver.restore(sess, './result/model'+str(cnn.num)+'.ckpt')  # 和之前保存的文件名一致,读取图
    # f = open('./result/CNN/CNNrecord.txt','a+')
    # f.write('第%d个训练器\n'%cnn.num)
    # f.write('训练器参数：卷积层数量%d 全连接层数量%d 卷积层步长%d 池化层步长%d 卷积层1大小%d 卷积层1输出通道%d 池化层1大小%d 卷积层2大小%d 卷积层2输出通道%d 池化层2大小%d '
    #         '全连接层1大小%d'%(cnn.convlayers,cnn.nnlayers,cnn.conv_patch,cnn.pool_patch,cnn.conv1,cnn.outchannal1,
    #                      cnn.pool1,cnn.conv2,cnn.outchannal2,cnn.pool2,cnn.nn1))
    #
    # try:
    #     i = 0
    #     cur_accuracy = 0
    #     while not coord.should_stop():
    #         image, label = sess.run([image_batch, label_batch])  # 在该处执行时弹出错误
    #         cur_accuracy += accuracy.eval(feed_dict={x: image, y: label, keep_prob: 1.0})
    #         i += 1
    #         print(i, cur_accuracy)
    # except tf.errors.OutOfRangeError:
    #     print('complete')
    #     print('测试正确率为:%f' % (cur_accuracy / i))
    # finally:
    #     coord.request_stop()  # 停止读入线程
    # coord.join(threads)
    # sess.close()
    # end_time = time.time()
    # print('程序运行耗时为：%.8s s' % (end_time - start_time))
