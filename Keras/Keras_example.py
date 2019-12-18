# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.contrib.eager as tfe
# import functools
import pathlib
import random
import os
import IPython.display as display
# import os

"""
keras训练简单神经网络
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#读取数据(直接将全部数据读出）
fashion_mnist = keras.datasets.fashion_mnist
(train_image,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
#查看输出数据
print(np.shape(train_image))
print(np.shape(train_labels))   #labels是一维数组
#进行均值归一化
train_image = train_image/255
test_images = test_images/255
#定义网络结构
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),  #变成二维张量m,28*28
    keras.layers.Dense(128,activation='relu'),  #三层网络层结构
    keras.layers.Dense(10,activation='softmax')
])
#网络编译（指明优化器，损失函数和度量方式)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])   #sparse_categorical_crossentropy不需要one_hot变量类型
#训练模型
model.fit(train_image, train_labels,epochs=10,verbose=2)    #verbose为是否打印信息

"""
模型的保存和回调
"""
# fashion_mnist = keras.datasets.fashion_mnist
# (train_image,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# train_image = train_image[:1000]/255.0
# train_labels = train_labels[:1000]  #取前1000个数据
# test_images = test_images/255.0
# def create_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape = (28,28)),  #变成二维张量m,28*28
#         keras.layers.Dense(128,activation='relu'),  #三层网络层结构
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(10,activation='softmax')
#     ])
#     #网络编译（指明优化器，损失函数和度量方式)
#     model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#     return model
# model = create_model()
# # model.summary() #可以显示各层的参数
# # #创建一个旨在训练期间保存权重的ModelCheckpoint
# checkpoint_path = 'result/example.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)    #显示文件夹路径
# print(checkpoint_path)
# # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)#verbose为信息保存模式
# # model.fit(train_image,train_labels,epochs=5,validation_data=(test_images,test_labels),callbacks=[cp_callback])#回调训练，每训练完一个轮回就保存一次
#
# # 读取模型
# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(test_images,test_labels,verbose=0)
# print('acc:%f,loss:%f'%(loss,acc))
"""
使用tensorflow读取csv文件格式的数据
重点：
1.实现csv读取文献的形成的ordered dict类型（数据集类别）向可用特征向量类型的转化
2.通过csv文件每一列的名称来进行索引
问题：
1. 官方例程中DenseFeature函数在该版本中不存在
"""
# tfe.enable_eager_execution()
# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
#
# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)  # 从指定链接下载图片，返回文件保存的路径，默认在~/.keras文件目录下


# 设置读取csv文件的方式（默认第一行为列名），使用dataset类实现增量式学习
# def getCsvFile(filepath):
#     LABEL_COLUMN = 'survived'  # 指定标签列
#     # COLUMN_NAMES = ['']   #默认第一行为列名字
#     dataset = tf.data.experimental.make_csv_dataset(
#         file_pattern=filepath,
#         batch_size=12,
#         label_name=LABEL_COLUMN,
#         na_value="?",
#         num_epochs=1
#     )
#     return dataset
# 读出数据
# raw_train_data = getCsvFile(train_file_path)  # ordered dict种类（有序字典？——像一个列表中存了很多个字典）
# raw_test_data = getCsvFile(test_file_path)
# # print(raw_train_data)
#
# examples, labels = next(
#     iter(raw_train_data))  # 通过迭代取出里面的数据(一个batch中的数据）创建可迭代对象(expample按字典读出，因为每一列都有对应的列的名称）——对应的名称下保存着对应的张量
# print(examples['sex'])
# print(np.shape(labels))
#
# # 数据预处理(创建特征向量）——输入：ordered dict类型 输出：非连续特征和连续特征组成的特征向量组合
# # 1. 创建类名指示器（非数值型连续特征）
# CATEGORIES = {
#     'sex': ['male', 'female'],
#     'class' : ['First', 'Second', 'Third'],
#     'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
#     'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
#     'alone' : ['y', 'n']
# }
# categorical_columns = []
# for feature,vocab in CATEGORIES.items():
#     cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab) #创建特征列
#     categorical_columns.append(tf.feature_column.indicator_column(cat_col)) #创建指示列并放入总列表中
#
# # 2. 连续数据标准化(每一列的数据进行均值归一化)
# def process_continuous_data(mean,data):
#     data = tf.cast(data,tf.float32)*1/(2*mean)
#     return tf.reshape(data,[-1,1])  #返回二维张量（本来是一维的特征向量）
# #创建数值列集和
# MEANS = {
#     'age' : 29.631308,
#     'n_siblings_spouses' : 0.545455,
#     'parch' : 0.379585,
#     'fare' : 34.385399
# }   #数值列字典
# numerical_columns = []
# for feature in MEANS.keys():
#     num_col = tf.feature_column.numeric_column(feature,normalizer_fn=functools.partial(process_continuous_data,MEANS[feature]))
#     numerical_columns.append(num_col)   #生成feature column
# # 对于数据全部已知数据的预处理
#
# # 创建预处理层
# preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numerical_columns)   #将原始数据转换为需要的特征数据
# # 产生问题——没有对应的DenseFeatures函数！
#
# # 创建模型
# model = tf.keras.Sequential([
#   preprocessing_layer,  #在这里放入数据预处理层（只需要将处理好的数据送入就可以了）（将预处理层加入进了网络中）——将输入数据从order dict类型转换为特征向量类型
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid'),
# ])
#
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])
#
# train_data = raw_train_data.shuffle(500) #表示混乱程度
# test_data = raw_test_data   #test_data不处理？
#
# model.fit(train_data, epochs=20)    #epoch为训练的总轮数   #fit函数会自动打印训练效果
# tess_loss, tess_accuracy = model.evaluate(test_data)    # 也会自动打印训练效果
"""
用tf.data加载图片（构建输入管道）
阅读：
1.python字典和列表之间的互相转换：
    当列表中每一个子列表的个数为2时，就可以转换为字典，也就是说前一个元素为key，后一个元素为value
    字典转换为列表同理，但是变成无顺序的二维列表

步骤：
1. 构建样本文件名列表
2. 构建样本文件标签列表
3. 定义图像预处理函数（读取，解码，裁剪，归一化）
4. 构造batch(shuffle,repeat等步骤）
5. 构造网络

训练数据的基本方法:
要使用此数据集训练模型，你将会想要数据：
    被充分打乱。
    被分割为 batch。
    永远重复。
    尽快提供 batch
"""
# # AUTOTUNE = tf.data.experimental.AUTOTUNE
# data_root_orig = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#     fname='flower_photos', untar=True)
# data_root = pathlib.Path(data_root_orig)  # 转换为一个path对象！牛逼！

# # for i in data_root.iterdir():  # 列举自己路径下的文件夹
# #     print(i)

# # 获得文件名队列
# all_image_paths = list(data_root.glob('*/*'))  # 找到该种格式的文件（这里是path对象组成的列表）
# all_image_paths = [str(path) for path in all_image_paths]  # 牛逼啊！将所有的路径都提取出来（和上一个有什么区别呢？这里转换为字符串对象）
# random.shuffle(all_image_paths)  # 读到的所有图片的次序
# # t1 = [parents_name.parent.name for parents_name in all_image_paths]
# # print(t1)
# image_count = len(all_image_paths)  # 图片个数

# # attributions = (data_root / 'LICENSE.txt').open(encoding='utf-8').readlines()[4:]  # 使用path类来直接open
# # attributions = [line.split(' CC-BY') for line in attributions]
# # attributions = dict(attributions)  # 列表转换为字典？当每一子列表中个数为2时

# # 列出可用标签
# label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir()) #获取路径下的文件夹名字（sorted为排序操作)
# # 标签分配索引（使用字典形式来生成索引)
# label_to_index = dict((name,index) for index,name in enumerate(label_names))
# # 为每个样本分配标签
# all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths] #寻找父类文件名称（上一级文件夹）

# # 加载和格式化图片
# # 使用io类读取数据
# def preprocess_image(img_raw):
#     img_tensor = tf.image.decode_image(img_raw)  # 解码成张量形式
#     img_final = tf.image.random_crop(img_tensor, [128, 128,3])  # 随即裁剪
#     # print(img_final.shape)
#     img_final = img_final / 255  # 归一化至[0,1]
#     return img_final

# def load_and_preprocess_image(img_path):
#     img_raw = tf.io.read_file(img_path)
#     return preprocess_image(img_raw)

# # 构建图片数据集
# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)   #构造字符串数据集（dataset)——属于tensorflow中的特殊的一类
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.contrib.data.AUTOTUNE) #数据集映射
# # 构建标签数据集
# label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)
# # 打包成(图片，标签)数据集
# image_label_ds = tf.data.Dataset.zip((image_ds,label_ds))   #打包成一个元组
# # print(image_label_ds)

# #或者直接将输入的文件名队列和文件标签队列切片
# # ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
# # def load_and_preprocess_from_path_label(path, label):
# #   return load_and_preprocess_image(path), label
# #
# # image_label_ds = ds.map(load_and_preprocess_from_path_label)

# # 构造batch
# BATCH_SIZE = 32
# # 设置一个和数据集大小一致的 shuffle buffer size(随机缓冲区）来保证数据被打乱
# ds = image_label_ds.shuffle(buffer_size=image_count)    #随机打乱数据
# ds = ds.repeat()    #可无限重复数据集
# ds = ds.batch(BATCH_SIZE)
# # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch
# # ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)  #从随即缓冲区中拿取元素
# # print(ds)

# # 符合预定输出
# def change_range(image,label):  #转换为[-1,1]
#   return 2*image-1, tf.cast(label,tf.int64)     #程序要求转换为int64类型
# keras_ds = ds.map(change_range) #batch经过映射

# # 构造网络
# mobile_net = tf.keras.applications.MobileNetV2(input_shape=(128,128,3),include_top=False)   #构造网络
# mobile_net.trainable = False

# # for i in range(3):
# #     image_batch, label_batch = next(iter(keras_ds)) #取出第一轮
# #     print(label_batch)
# # feature_batch = mobile_net(image_batch)
# # print(feature_batch)

# # 创建模型
# model = tf.keras.Sequential([
#     mobile_net,
#     tf.keras.layers.GlobalAveragePooling2D(),   #转到全连接层不需要重塑张量维度？
#     tf.keras.layers.Dense(len(label_names),activation='softmax')
# ])
# model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
#               loss='sparse_categorical_crossentropy',
#               metrics=["accuracy"])

# # 将通道与训练器结合
# # steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE)
# # print(steps_per_epoch)
model.fit(keras_ds,epochs=5,steps_per_epoch=3)













