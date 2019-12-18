# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-12

"""
使用预训练好的MobileV2网络来提取图片的特征,输入图片的大小需要为[96, 128, 160, 192, 224]
样本最小图片大小：
    height:231
    weight:271

Feature extraction:
    constructMobileV2_untrained:只训练最后一个分类器(top_layer===>也就是the last layer全连接层)
Fine_tuning:
    constructMobileV2_FineTuning:设置fine_tune_at作为开始训练的层数（原来的mobileV2一共有155层）


步骤：
    1. 导入batch（创建数据输入通道）——划分为三层（train,valid,test)
    2. 创建模型（预训练好的卷积层模型参数）
    3. 训练模型
    4. 记录输出
    5. 保存训练模型

问题：
    1. 如何取出CNN网络中间层的网络输出？(取出训练数据）
        使用函数模型API，新建一个model，将输入和输出定义为原来的model的输入和想要的那一层（如何取出确定层？使用model.summary()打印出CNN每层名字，使用对应名字取出）的输出，然后重新进行predict
"""
import tensorflow as tf
import Data_ReadImage as ReadImage
import numpy as np
import os 

# 裁剪掉一部分网络模型（产生过拟合现象）
def constructPartMobileV2(Image_height, Image_weight, classnum,cut_at):
    #加载预训练的模型
    base_v2model = tf.keras.applications.MobileNetV2(input_shape=(Image_weight,Image_height,3), include_top=False, weights = 'imagenet')
    base_v2model.trainable = False  #变为从网络中间层提取特征，减少训练参数
    #尝试取出一部分作为训练
    cut_layer = base_v2model.layers[cut_at]
    #生成裁剪的模型
    partMobileV2model = tf.keras.models.Model(inputs=base_v2model.input,outputs=cut_layer.output)
    for current_layer in partMobileV2model.layers[:cut_at]:
        current_layer.trainable = False             
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # 特征降维
    prediction_layer = tf.keras.layers.Dense(classnum, activation='softmax')  # 输出为相应的分类
    model = tf.keras.Sequential([
        partMobileV2model,
        global_average_layer,
        prediction_layer
    ])
    base_learningrate = 0.0001
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=base_learningrate),
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    print('建模完成')
    return model

# 构造MobileV2模型（feature extraction)
def constructMobileV2_FeatureExtraction(Image_height, Image_weight, classnum):
    # 加载预训练模型
    base_v2model = tf.keras.applications.MobileNetV2(input_shape=(Image_weight, Image_height, 3), include_top=False,
                                                     weights='imagenet')  # 因为是第一层，所以要指定输入图片的维度
    base_v2model.trainable = False  # 冻结其余层
    # 加入可训练自己数据的顶层结构
    # maxpool_layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)    #特征维度降维
    # add_layer = tf.keras.layers.Conv2D(1560,(1,1),activation='relu')  #特征通道降维
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # 特征降维
    prediction_layer = tf.keras.layers.Dense(classnum, activation='softmax')  # 输出为相应的分类
    # 建模
    model = tf.keras.Sequential([
        base_v2model,
        # add_layer,
        global_average_layer,
        prediction_layer
    ])
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=base_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('模型构造完成')
    return model


# Fine_tuning
def constructMobileV2_FineTuning(image_height, image_weight, classnum):
    # 加载预训练模型
    base_v2model = tf.keras.applications.MobileNetV2(input_shape=(image_weight, image_height, 3), include_top=False,
                                                     weights='imagenet')
    base_v2model.trainable = True  # 训练顶部的几层，越靠近顶部神经网络层越具有指向性（符合对应的数据集)
    # 设置开始训练的层数
    fine_tune_at = 130
    # 将该层之前的层数设置为不能训练
    for layer in base_v2model.layers[:fine_tune_at]:
        layer.trainable = False
    # 加入顶层结构，使其适合对应的数据集
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # 全局降维
    prediction_layer = tf.keras.layers.Dense(classnum, activation='softmax')
    # 建模
    model = tf.keras.Sequential([
        base_v2model,
        global_average_layer,
        prediction_layer
    ])
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=base_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    print('建图完成')
    return model


if __name__ == '__main__':
    # 使用cpu进行训练
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # 定义一些常数
    image_weight = 224  #修改图片的大小时必须修改Data_ReadImage的preProcessImage函数
    image_height = 224
    classnum = 47
    batchsize = 16
    # dirpath = 'C:/Users/1/Desktop/LBP/temp'  # 存放图片文件夹的主路劲
    dirpath = '../images'   #服务器图片地址
    cut_at = 100
    epochs = 100
    step_per_epochs = 212
    # 构造数据集
    ds_train, ds_valid = ReadImage.getDataSet(dirpath, batchsize)
    # 加载预训练模型
    model = constructMobileV2_FeatureExtraction(image_height, image_weight, classnum)
    model.summary() #打印网络结构
    # 训练过程
    # # 使用tensorboard记录模型
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./graph',
                                                write_graph=True, write_images=True
                                                )
    # 使用ModelCheckpoint保存模型
    checkpoint_path = 'result/MobileV2_Feature_60_224_reduce.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,save_best_only=True)
    # 训练模型
    print('进入训练')
    model.fit(ds_train, epochs=epochs, steps_per_epoch=step_per_epochs,validation_data=ds_valid,validation_steps=6,callbacks=[tbCallBack])

    # # 测试过程
    # model.load_weights(checkpoint_path)
    # test_image,test_labels = ReadImage.getTestDateSet('cnn')
    # feature_model = tf.keras.models.Model(inputs = model.input, outputs=model.get_layer('global_average_pooling2d').output) #从CNN层数名命中取出参数
    # feature_output = feature_model.predict(test_image)
    # print(np.shape(feature_output))
    # loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    # print(loss)
    # print(acc)