# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-12-13

"""
使用预训练好的深度特征去实现Bag of words的训练

步骤：
    1.确定选用好的CNN模型（模型的特征向量是否需要降低数量？）
    2.模型赋值取出，提取出特征产生层，得到深度特征
    3.对特征进行维度变换（将二维特征变成一维向量形式）
    4.将处理后的特征放入KMEANS聚类器中进行聚类
    5.得到码本，对图像进行histogram编码
    6.送入svm中实现样本分类

总训练步骤：
    先提取分别特征(train,test)，再进行分类

问题：
    1.是否需要减少样本特征数量
        尝试使用不同特征数量（输出channal数量不同）进行训练
    2.如何将二维特征平铺，转换为特征向量
        双重循环切片操作
    3.如何对应好编码过程（不用shuffle了，一次读取出全部的变量）
        sklearn不需要onehot标签
    4.SVM训练的码本顺序打乱？
        原来的标签已经为乱序
    5.图片样本的特征数量太多，聚类时间太慢！
        5.1 尝试第一次降维（输出特征数量为300）——生成的特征集和测试集差别特别大
    6.使用svm代替全连接层
        训练准确度非常高，但是测试准确度相对较低
        6.1 加入降维手段（怀疑过拟合，特征数量太大） PCA,KernalPca

"""
import MobileV2_FeatureDetection as V2
import tensorflow as tf
import numpy as np
import Data_ReadImage as ReadImage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import KernelPCA,PCA
from sklearn import svm,multiclass
import joblib
import CsvFunction as CSV
import os
import time

# 取出预训练好的特征输出层模型（每次提取模型之前需要重新看模型名称）
def getPre_trainedModel(checkpoint_path, image_height, image_weight):
    # pre_trained_model = V2.constructMobileV2_FeatureExtraction(image_height, image_weight, 47)  # 这里不重要，重要的是读取数据
    # pre_trained_model.load_weights(checkpoint_path)  # 这里很重要，取哪个参数
    # # pre_trained_model.summary()
    # feature_extracted_model = tf.keras.models.Model(inputs=pre_trained_model.input,
    #                                                 outputs=pre_trained_model.get_layer('conv2d').output)  # 得到特征层(特征层名字可能需要改一改(根据是否需要降维特征）)
    # 这里直接使用mobileV2去提取特征(特征维度过高),加入全局池化,送入svm
    base_v2model = tf.keras.applications.MobileNetV2(input_shape=(image_height,image_weight,3),include_top=False,weights = 'imagenet')   # 因为是第一层，所以要指定输入图片的维度
    base_v2model.trainable = False  # 冻结其余层
    model = tf.keras.Sequential([
        base_v2model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    feature_extracted_model = model  #直接使用mobileV2去提取特征
    feature_extracted_model.summary()
    return feature_extracted_model


# 对特征进行切片操作,返回特征向量
def getFeatureVector(cur_image_feature):
    """
    双重循环进行数组切分
    :param cur_image_feature: 卷积层输出的4维数组 (n,h,w,channal)
    :return: 每个数组对应的特征向量
    """
    # print(np.shape(cur_image_feature))
    batch_size, w, h, channal = np.shape(cur_image_feature)
    cur_feature_vector = []
    for i in range(batch_size):
        for j in range(channal):
            cur_feature = cur_image_feature[i, :, :, j].flatten()
            cur_feature_vector.append(cur_feature)
    return cur_feature_vector

# 设置增量式学习函数(ex_epoches增量式学习的训练次数
def partialAddingLearn(feature_extracted_model,n_cluster, channal, ex_epoches,batch_size=141, epoches=32):
    # 获得训练数据
    Kmeans_ds = ReadImage.getKmeansDataSet(batch_size)
    print('数据提取完成')
    # 获得预训练模型
    # feature_extracted_model = getPre_trainedModel(checkpoint_path, image_height, image_weight)
    # print('预训练模型提取完成')
    # 使用tensorflow每次训练数据
    iterator = Kmeans_ds.make_initializable_iterator()
    data_element = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    Kmeans_label = []   #增量式学习不需要生成全体样本特征
    Kmeans_feature = []
    print('进入训练特征提取+增量式kmeans学习过程')
    kmeans_mode = MiniBatchKMeans(n_clusters=n_cluster,batch_size=batch_size*channal,random_state=0) #初始化增量式学习模型
    for cur_epoch in range(ex_epoches):
        for i in range(epoches):
            Kmeans_image, curKmeans_label = sess.run(data_element)
            cur_image_feature = feature_extracted_model.predict(Kmeans_image)   #得到一个batchsize的深度特征2维矩阵，可以放入增量式学习当中
            cur_image_vectors = getFeatureVector(cur_image_feature) #转为一维变量
            if cur_epoch==0:    #只有在第一次循环时收集样本信息
                Kmeans_feature.extend(cur_image_vectors)
                Kmeans_label.extend(curKmeans_label)
            kmeans_mode.partial_fit(cur_image_vectors)
            print('第%d轮增量式学习完成'%i)
    sess.close()
    # 保存kmeans模型
    Kmeans_feature = np.asarray(Kmeans_feature)
    Kmeans_label = np.asarray(Kmeans_label)
    CSV.csvWrite('./BoW/data_csv/train_label.csv', enumerate(Kmeans_label))
    joblib.dump(kmeans_mode, filename='./BoW/result/kmeans_' + str(n_cluster) + '_.pkl') # 保存文件
    print('增量式学习完成,kmeans——%d保存完成'%n_cluster)
    # 获得相应的特征标签
    print('获得相应的特征标签')
    feature_labels = kmeans_mode.predict(Kmeans_feature)
    print('特征标签获取完成')
    # 进入特征编码过程
    print('进入特征编码过程')
    m = len(Kmeans_label)   #样本数
    print(m)
    histogram_code = []
    for cur_image in range(m):
        cur_code = np.zeros((1, n_cluster))
        for cur_feature in range(cur_image * channal, (cur_image + 1) * channal):
            cur_cluster = feature_labels[cur_feature]
            cur_code[0, cur_cluster] += 1
        cur_code = cur_code / np.sum(cur_code)  # 归一化
        histogram_code.append(cur_code[0])  #数组只要在创造后就变为两维（ones,zeros)
    histogram_code = np.asarray(histogram_code)
    print(np.shape(histogram_code))
    print('训练集histogram编码完成')
    CSV.csvWrite('./BoW/data_csv/train_code_' + str(n_cluster) + '.csv', histogram_code)
    print('直方图编码完成')
    return kmeans_mode,histogram_code,Kmeans_label

# 对训练集进行前向传播,得到深度特征，将深度特征进行降维后放入kmeans学习器中，最后保存Kmeans模型
def featureExtract(checkpoint_path, image_height, image_weight, batch_size=141, epoches=32):
    # 获得训练数据
    Kmeans_ds = ReadImage.getKmeansDataSet(batch_size)
    print('数据提取完成')
    # 获得预训练模型
    feature_extracted_model = getPre_trainedModel(checkpoint_path, image_height, image_weight)
    print('预训练模型提取完成')
    # 使用tensorflow每次训练数据
    iterator = Kmeans_ds.make_initializable_iterator()
    data_element = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    Kmeans_feature = []
    Kmeans_label = []
    print('进入训练集特征提取')
    for i in range(epoches):  # 读取训练集数据
        Kmeans_image, cur_Kmeans_label = sess.run(data_element)
        cur_image_feature = feature_extracted_model.predict(Kmeans_image)
        Kmeans_feature.extend(getFeatureVector(cur_image_feature))
        Kmeans_label.extend(cur_Kmeans_label)
    sess.close()
    Kmeans_feature = np.asarray(Kmeans_feature)
    Kmeans_label = np.asarray(Kmeans_label)
    print('Kmeans特征数量为：数量:%d  维度:%d' % np.shape(Kmeans_feature))
    print('Kmeans标签数量为：%d（样本数量） ' % np.shape(Kmeans_label))
    print('特征提取完成')
    CSV.csvWrite('./BoW/data_csv/train_label.csv', enumerate(Kmeans_label))
    return Kmeans_feature, Kmeans_label


# 初始化Kmeans学习器
def learnKmeans(Kmeans_feature, Kmeans_label, n_cluster, channal):
    print('进入kmeans训练')
    kmeans = MiniBatchKMeans(n_clusters=n_cluster, batch_size=int(channal*100), random_state=0, verbose=1).fit(Kmeans_feature)   #打印一些输出
    joblib.dump(kmeans, filename='./BoW/result/kmeans_' + str(n_cluster) + '_.pkl')  # 保存文件
    print('kmeans文件保存完成，同时写入完成')
    # 进入直方图编码（可以直接用kmeans.labels_获得kmeans聚类样本的簇分类
    m = np.shape(Kmeans_label)[0]  # 得到样本数量
    histogram_code = []
    feature_labels = kmeans.labels_  # 获得对应标签
    for cur_image in range(m):
        cur_code = np.zeros((1, n_cluster))
        for cur_feature in range(cur_image * channal, (cur_image + 1) * channal):
            cur_cluster = feature_labels[cur_feature]
            cur_code[0, cur_cluster] += 1
        cur_code = cur_code / np.mean(cur_code)  # 归一化
        histogram_code.append(cur_code[0])  #数组只要在创造后就变为两维（ones,zeros)
    histogram_code = np.asarray(histogram_code)
    print(np.shape(histogram_code))
    print('训练集histogram编码完成')
    CSV.csvWrite('./BoW/data_csv/train_code_' + str(n_cluster) + '.csv', histogram_code)
    f = open('./BoW/txt_record/kmeansRecord.txt', 'a+')
    f.write('特征维度为:%d \n' % n_cluster)
    f.write('SSE系数为:%f \r\n' % kmeans.inertia_)
    f.close()
    return histogram_code,kmeans


# 得到测试集数据
def getTestHistogram(kmeans, feature_extracted_model, n_cluster):
    # 获得测试数据
    test_image, test_label = ReadImage.getTestDateSet('bow')
    print('采集数据完成')
    # 获得模型
    # feature_extracted_model = getPre_trainedModel(checkpoint_path, image_height, image_weight)
    # print('预训练模型提取完成')
    test_image_feature = feature_extracted_model.predict(test_image)
    m, w, h, channal = np.shape(test_image_feature) #在经过网络后再取通道数！
    test_feature_vector = getFeatureVector(test_image_feature)
    test_feature_vector = np.asarray(test_feature_vector)
    print('特征向量处理完成 数量:%d  维度:%d'%np.shape(test_feature_vector))
    # 聚类
    vector_label = kmeans.predict(test_feature_vector)
    print('聚类提取完成')
    # 编码
    test_histogram_code = []
    for cur_image in range(m):
        cur_feature_vector = np.zeros((1, n_cluster))
        for cur_channal in range(cur_image * channal, (cur_image + 1) * channal):
            cur_feature_label = vector_label[cur_channal]
            cur_feature_vector[0, cur_feature_label] += 1
        cur_feature_vector = cur_feature_vector / np.sum(cur_feature_vector)    #进行总和归一化
        test_histogram_code.append(cur_feature_vector[0])
    test_histogram_code = np.asarray(test_histogram_code)
    print('测试集histogram编码完成')
    CSV.csvWrite('./BoW/data_csv/test_code_' + str(n_cluster) + '.csv', test_histogram_code)
    CSV.csvWrite('./BoW/data_csv/test_label.csv', enumerate(test_label))
    return test_histogram_code, test_label


# SVM训练(SVM训练不需要onehot标签）+ 同时生成测试准确率
def learnSVM(C, train_code, train_label, test_code, test_label):
    n = np.shape(train_code)[1]  # 获得n_cluster数量
    print('进入SVM训练')
    svc_classifier = svm.SVC(C=C, kernel='rbf', random_state=0,gamma='auto')
    model = multiclass.OneVsOneClassifier(svc_classifier, -1)
    clf = model.fit(train_code, train_label)
    joblib.dump(clf, filename='./BoW/result/svm_' + str(n) + '_' + str(C) + '_.pkl')
    print('训练完成')
    train_accuracy = clf.score(train_code, train_label)
    print('train_accuracy:%f' % train_accuracy)
    test_accuracy = clf.score(test_code, test_label)
    print('test_accuracy:%f' % test_accuracy)
    print('训练完成')
    f = open('./BoW/txt_record/svmRecord.txt', 'a+')
    f.write('特征维度为:%d \n' % n)
    f.write('惩罚系数为:%f \n' % C)
    f.write('训练准确率: %.4f 测试准确率:%.4f\n\r' % (train_accuracy, test_accuracy))
    f.close()

# 使用svm当作最后的全连接层
def SVM_FullConnected(pretrained_model, C, d_, batch_size=141, epoches=32):
    start_time = time.time()
    #制造训练数据通道
    train_ds = ReadImage.getKmeansDataSet(batch_size)
    print('训练通道构建完成')
    iterator = train_ds.make_initializable_iterator()
    data_element = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    train_label = []  # 增量式学习不需要生成全体样本特征
    train_data = []
    for i in range(epoches):
        cur_train_image,cur_train_label = sess.run(data_element)
        cur_train_data = pretrained_model.predict(cur_train_image)  #svm使用的是均值池化后的输出向量
        train_data.extend(cur_train_data)
        train_label.extend(cur_train_label)
    print('训练数据提取完成。训练数据维度%d %d'%np.shape(train_data))
    sess.close()
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    #得到测试数据
    test_image, test_label = ReadImage.getTestDateSet('bow')
    print('采集测试数据完成')
    test_data = pretrained_model.predict(test_image)
    print('测试数据提取完成，测试数据维度为%d %d'%np.shape(test_data))
    end_time1 = time.time()
    print('数据提取用时:%.8ss'%(end_time1-start_time))
    #将训练数据和测试数据进行降维处理
    pca_model = PCA(n_components=d_)
    pca_model = KernelPCA(n_components=d_,kernel='rbf') #使用rbf降维方式
    pca_model.fit(X=train_data)
    train_data_d_ = pca_model.transform(train_data)
    print('降维后的训练数据集维度为%d %d'%np.shape(train_data_d_))
    test_data_d_ = pca_model.transform(test_data)
    print('降维后的训练数据集维度为%d %d'%np.shape(test_data_d_))
    d_time = time.time()
    print('降维时间:%.8ss'%(d_time-end_time1))
    #将训练数据核测试数据输入svm中
    svc_classifier = svm.SVC(C=C, kernel='rbf', random_state=0, gamma='auto')
    model = multiclass.OneVsOneClassifier(svc_classifier, -1)
    clf = model.fit(train_data_d_, train_label)
    end_time2 = time.time()
    print('训练完成,用时%.8ss'%(end_time2-d_time))
    train_accuracy = clf.score(train_data_d_,train_label)
    end_time3 = time.time()
    print('tran_accuracy:%f,用时%.8s s'%(train_accuracy,end_time3-end_time2))
    test_accuracy = clf.score(test_data_d_,test_label)
    end_time4 = time.time()
    print('test_accuracy:%f,用时%.8s s'%(test_accuracy,end_time4-end_time3))
    print('训练完成,总用时%.8s s '%(end_time4-start_time))
    f = open('./BoW/txt_record/svmFCRecord.txt', 'a+')
    f.write('惩罚系数为:%f \n' % C)
    # f.write('降维指标为：%d 降维方式：rbf内核\n'%d_)
    f.write('one vs one classfier')
    f.write('训练准确率: %.4f 测试准确率:%.4f\n\r' % (train_accuracy, test_accuracy))
    f.close()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    checkpoint_path = 'result/MobileV2_Feature_60_224_reduce.ckpt'  #使用降维后的输出
    image_height = 224
    image_weight = 224  #现在默认运行高纬度
    channal = 300  # 使用降维后的网络
    # 测试用数据
    # batch_size = 43
    # epoches = 5
    ex_epoches= 2
    n_cluster = 200
    C = 3000
    d_ = 650 #降维维度
    # 进入bagofWord训练
    # Kmeans_feature,train_label = featureExtract(checkpoint_path, image_height, image_weight)
    # train_code,kmeans = learnKmeans(Kmeans_feature,train_label,n_cluster,channal)   #无标签学习
    # 增量式学习方式
    # feature_extracted_model = getPre_trainedModel(checkpoint_path, image_height, image_weight)
    # print('预训练采集层采集完成')
    # kmeans,train_code,train_label = partialAddingLearn(feature_extracted_model,n_cluster,channal,ex_epoches)
    # test_code,test_label = getTestHistogram(kmeans, feature_extracted_model,n_cluster)
    # learnSVM(C,train_code,train_label,test_code,test_label)
    # print('训练完成')
    # 使用svm代替最后全连接层
    feature_extracted_model = getPre_trainedModel(checkpoint_path, image_height, image_weight)
    print('预训练模型采集成功')
    # for C in range(10,201,10):
    #     for d_ in range(650,1101,50):
    print('C:'+str(C))
    print('d_:'+str(d_))
    SVM_FullConnected(feature_extracted_model,C,d_)
