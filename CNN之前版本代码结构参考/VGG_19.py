# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-08

"""
模仿VGG-19网络
INPUT：112*112*1
classnum=47

问题：
1.weights_regularizer=tf.truncated_normal_initializer(stddev=0.1)会报错——TypeError: Value passed to parameter 'shape' has DataType float32 not in list of allowed values: int32, int64
2.新版本tensorflow不能使用start_queue_runners函数（必须使用tf.data函数来实现数据的读取）
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import CsvFunction as CSV
import TF_readImage as TF_read
import numpy as np
import os

# # 定义简单的函数产生截断的正态分布
# trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# 定义函数 inception_v3_arg_scope 用来生成网络中经常用到的函数的默认参数
def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1,
                           batch_norm_var_collection="moving_vars"):
    batch_norm_params = {
        "decay": 0.9997, "epsilon": 0.001, "updates_collections": tf.GraphKeys.UPDATE_OPS,
        "variables_collections": {
            "beta": None, "gamma": None, "moving_mean": [batch_norm_var_collection],
            "moving_variance": [batch_norm_var_collection]
        }
    }  # batch正则化

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=tf.nn.relu):
        # 对卷积层生成函数的几个参数赋予默认值（BN算法的卷积层）
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',  # 默认为填充0
                            #weights_regularizer=tf.truncated_normal_initializer(stddev=stddev),     #这句话有点问题！！！
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            stride=1):
                            # 在BN算法函数（Batch Normalization)BN方法对每一个mini-batch数据内部正则化处理，使输出规范到N(0,1)中
                            # normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as scope:
                return scope  # 定义作用的定义域
# 定义VGG-19网络
def VGG_19(inputs, scope=None):
    end_points = {}
    with slim.arg_scope([slim.conv2d],padding='SAME',
                        activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,stride=1):
        with tf.variable_scope(scope, 'VGG_19', [inputs]):
            net1 = slim.conv2d(inputs, num_outputs=128, kernel_size=[3, 3], scope='conv1_1')
            print(np.shape(net1))
            net1 = slim.conv2d(net1, num_outputs=128, kernel_size=[3, 3], scope='conv1_2')
            pool1 = slim.max_pool2d(net1, kernel_size=[3, 3], stride=2, scope='conv1_pool')

            net2 = slim.conv2d(pool1, 256, kernel_size=[3, 3], scope='conv2_1')
            net2 = slim.conv2d(net2, 256, kernel_size=[3, 3], scope='conv2_2')
            net2 = slim.conv2d(net2, 256, kernel_size=[3, 3], scope='conv2_3')
            net2 = slim.conv2d(net2, 256, kernel_size=[3, 3], scope='conv2_4')
            pool2 = slim.max_pool2d(net2, kernel_size=[3, 3], scope='conv2_pool')

            net3 = slim.conv2d(pool2, 512, kernel_size=[3, 3], scope='conv3_1')
            net3 = slim.conv2d(net3, 512, kernel_size=[3, 3], scope='conv3_2')
            net3 = slim.conv2d(net3, 512, kernel_size=[3, 3], scope='conv3_3')
            net3 = slim.conv2d(net3, 512, kernel_size=[3, 3],scope='conv3_4')
            pool3 = slim.max_pool2d(net3, kernel_size=[3, 3], scope='conv3_pool')

            net4 = slim.conv2d(pool3, 512, kernel_size=[3, 3], scope='conv4_1')
            net4 = slim.conv2d(net4, 512, kernel_size=[3, 3], scope='conv4_2')
            net4 = slim.conv2d(net4, 512, kernel_size=[3, 3], scope='conv4_3')
            net4 = slim.conv2d(net4, 512, kernel_size=[3, 3], scope='conv4_4')
            pool4 = slim.max_pool2d(net4, kernel_size=[3, 3], scope='conv4_pool')
            pool3_flat = tf.reshape(pool4,[-1,7*7*512])

            fc1 = slim.fully_connected(pool3_flat, 1024, scope='fc1')
            fc1_dropout = tf.nn.dropout(fc1,0.8)
            fc2 = slim.fully_connected(fc1_dropout, 1024, scope='fc2')
            fc2_dropout = tf.nn.dropout(fc2,0.8)
            fc3 = slim.fully_connected(fc2_dropout, 47, scope='fc3')

            end_points["Logits"] = fc3
            end_points["prediction"] = slim.softmax(fc3)

            return end_points

if __name__=='__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    start_time = time.time()
    #定义常用信息
    batchsize = 16
    outputsize = 47
    image_height = 112
    image_weight = 112
    epochs = 6  # 数据增强
    # 读取样本信息
    trainfile_name = CSV.csvRead('train', 'name')
    trainfile_label_str = CSV.csvRead('train', 'label')
    trainfile_label_float = CSV.str2float(trainfile_label_str)
    trainlabel_num = CSV.np.shape(trainfile_label_str)[1]

    testfile_name = CSV.csvRead('test', 'name')
    testfile_label_str = CSV.csvRead('test', 'label')
    testfile_label_float = CSV.str2float(testfile_label_str)
    print('读取数据完成')
    # 构造batch
    image_batch, label_batch = TF_read.getBatch(trainfile_name, trainfile_label_float, batchsize, image_height,
                                                image_weight, num_epochs=epochs)  # 训练过程
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # 定义占位符
    x_ = tf.placeholder(tf.float32,[None,image_height,image_weight,1])
    y_ = tf.placeholder(tf.float32,[None,47])
    x_image = tf.reshape(x_,[-1,int(image_weight),int(image_height),int(1)])
    # 初始化图
    scope1 = inception_v3_arg_scope()
    with slim.arg_scope(scope1):    #返回定义好的scope
        end_points = VGG_19(x_image)
    # 获得正确率
    prediction = end_points['prediction']
    # 打印出标签值和真实值
    label_true = tf.argmax(y_,1)
    label_predict = tf.argmax(prediction,1)
    # 计算正确率
    correct_prediction = tf.equal(label_predict,label_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 定义损失函数
    logits = end_points['Logits']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=logits))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=prediction))
    
    # print(type(loss))
    # 定义优化器
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
    print('建图完成')
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
            # 
            # train_step.run(feed_dict={x_: image, y_: label})
            sess.run(train_step,feed_dict={x_: image, y_: label})
            train_accuracy, loss1,label_true1,label_predict1=sess.run([accuracy,loss,label_true,label_predict],feed_dict={x_: image, y_: label})

            # train_accuracy = accuracy.eval(
            #         feed_dict={x_: image, y_: label})  # feed不能为张量
            # loss1 = loss.eval(feed_dict={x_:image,y_:label})
            # label_true1 = label_true.eval(feed_dict={x_:image,y_:label})
            # label_predict1 = label_predict.eval(feed_dict={x_:image,y_:label})
            if i % 5 == 0:
                end_time = time.time()
                print('step:%d ' % i + "training accuracy：%f" % train_accuracy + 'loss: %f'%loss1)
                print('VGG_19运行耗时为：%.8s s' % (end_time - start_time))
                print(label_true1)
                print(label_predict1)
            # train_step.run(feed_dict={x_: image, y_: label})
            i += 1
            # 
            # train_step.run(feed_dict={x_: image, y_: label})
            # if i % 5 == 0:
            #     train_accuracy, loss1,label_true1,label_predict1,
            #     train_accuracy = accuracy.eval(
            #         feed_dict={x_: image, y_: label})  # feed不能为张量
            #     loss1 = loss.eval(feed_dict={x_:image,y_:label})
            #     label_true1 = label_true.eval(feed_dict={x_:image,y_:label})
            #     label_predict1 = label_predict.eval(feed_dict={x_:image,y_:label})
            #     end_time = time.time()
            #     print('step:%d ' % i + "training accuracy：%f" % train_accuracy + 'loss: %f'%loss1)
            #     print('VGG_19运行耗时为：%.8s s' % (end_time - start_time))
            #     print(label_true1)
            #     print(label_predict1)
            # # train_step.run(feed_dict={x_: image, y_: label})
            # i += 1
            


            # 
            
            
            
            # print(name)
    except tf.errors.OutOfRangeError:
        print('complete')
    finally:
        coord.request_stop()  # 停止读入线程
    coord.join(threads)  # 线程加入主线程，等待主线程结束
    # saver.save(sess, './model.ckpt')
    end_time = time.time()
    sess.close()
    print('VGG_19运行耗时为：%.8s s' % (end_time - start_time))



