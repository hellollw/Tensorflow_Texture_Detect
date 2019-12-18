CNN之前版本代码结构参考：
	tensorflow1.5版本编码，包含了如何最基本的定义卷积层，池化层，如何构建老版本的数据通道
	VGG19使用slim包构图，但是感觉slim包有点问题（建议不要用）
Keras：
	新版本tensorflow框架，使用keras包建图，使用data API构建数据通道。
	使用MobileV2网络
	实现CNN+SVM,CNN+词袋模型