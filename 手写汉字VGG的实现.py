import os
from PIL import Image
import numpy as np

"""
搭建1个类似VGG网络（参考手写数据集模型结构），实现手写汉字数据的分类问题。
要求：验证数据集准确率达到90%以上。 请提交 代码 + 验证准确率截图。

PIL 包整合到了pillow下面，需要你安装pillow。可以在anaconda环境中（windows下） 
执行   pip install pillow
"""

image_width = 50
image_height = 100

def fetch_X_and_Y(dir_path):
	"""
	基于给定的文件夹路径加载该文件夹下子文件夹中的手写汉字图片的数据，并返回X和Y
	:param dir_path:  文件夹路径，eg: "D:\中文字符识别\训练数据"或者"D:\中文字符识别\验证数据"
	:return:
	"""
	X = []
	Y = []
	# a. 获取子文件夹的名称
	dirs = os.listdir(dir_path)
	print(dirs)
	# b. 遍历所有子文件夹
	for dir_name in dirs:
		# 将子文件夹名称转换为标签纸
		label = int(dir_name)
		# b1. 构建子文件夹的路径
		c_dir_path = os.path.join(dir_path, dir_name)
		print("加载文件夹'{}'中的图像数据!!!".format(c_dir_path))
		# b2. 获取子文件夹中的图像文件的名称
		image_names = os.listdir(c_dir_path)
		# b3. 遍历所有图像，读取图像数据构建X和Y
		for image_name in image_names:
			# b31. 构建图像的路径
			image_path = os.path.join(c_dir_path, image_name)
			# b32. 加载图像
			img = Image.open(image_path)
			# b33. 将RGB的图像转换为灰度图像，因为汉字对于颜色是不敏感的
			img = img.convert("L")
			# b34. 由于原始图像的大小是不固定的，导致构建出来的特征向量也是大小不固定的，所以将图像转换为大小一致的情况
			img = img.resize((image_width, image_height))
			# b35. 将图像转换为numpy数组的形式
			img_arr = np.array(img).reshape(image_height, image_width, 1)
			# 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
			img_arr = img_arr / 255.0
			# b36. 将图像的特征属性数据添加到X和Y中
			X.append(img_arr)
			Y.append(label)
	# c. 将X和Y转换为numpy数组的形式
	X = np.asarray(X).astype(np.float64)
	Y = np.asarray(Y).astype(np.float64)
	return X, Y


# b. 训练数据的产生/获取（基于numpy随机产生<可以先考虑一个固定的数据集>）
x_train, y_train = fetch_X_and_Y(dir_path='ChineseChar/train')
total_train_samples = np.shape(x_train)[0]
x_test, y_test = fetch_X_and_Y(dir_path='ChineseChar/valid')
print("数据加载完成，进行模型训练!")
print("训练数据格式:{} -- {}".format(np.shape(x_train), np.shape(y_train)))
print("测试数据格式:{}".format(np.shape(x_test)))
print(y_train)

import tensorflow as tf

# learning_rate=0.001
# epoches=10
# #dropout保留节点
# keep_prob=0.5
#
# def conv_op()

#VGG11
my_graph=tf.Graph()
with my_graph.as_default():
	weights={
		'conv1_1':tf.Variable(tf.truncated_normal([3,3,1,64],stddev=0.1)),
		'conv2_1':tf.Variable(tf.truncated_normal([3,3,64,64],stddev=0.1)),
		'conv3_1': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
		'conv3_2': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)),
		'conv4_1': tf.Variable(tf.truncated_normal([3, 3, 128, 512], stddev=0.1)),
		'conv4_2': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
		'conv5_1': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1)),
		'conv5_2': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.1))
	}
	biases = {
		'conv1_1': tf.Variable(tf.constant(0.1, shape=[64])),
		'conv2_1': tf.Variable(tf.constant(0.1, shape=[64])),
		'conv3_1': tf.Variable(tf.constant(0.1, shape=[128])),
		'conv3_2': tf.Variable(tf.constant(0.1, shape=[128])),
		'conv4_1': tf.Variable(tf.constant(0.1, shape=[512])),
		'conv4_2': tf.Variable(tf.constant(0.1, shape=[512])),
		'conv5_1': tf.Variable(tf.constant(0.1, shape=[512])),
		'conv5_2': tf.Variable(tf.constant(0.1, shape=[512]))
	}

with my_graph.as_default():
	x=tf.placeholder(tf.float32,shape=[None,100,50,1],name='x')
	y=tf.placeholder(tf.float32,shape=[None,10],name='y')
	training=tf.placeholder(tf.bool,name='training')
			#dropout保留节点
	keep_prob=tf.placeholder(tf.float32,name='keep_prob')
			#初始化学习率
	initial_learning_rate=0.00001
			#正则化因子
	weight_decay=1e-4

def conv_op(input,filter_w,bias,strides=1):
	conv=tf.nn.conv2d(input=input,filter=filter_w,strides=[1,strides,strides,1],padding='SAME')
	conv=tf.nn.bias_add(conv,bias)
	conv=tf.nn.relu6(conv)
	return conv

def max_pool(input,k=2):
	ksize=[1,k,k,1]
	strides=[1,k,k,1]
	pool_out=tf.nn.max_pool(value=input,ksize=ksize,strides=strides,padding='SAME')
	return pool_out

def fc(input,flatten_shape,fc_shape):
	weights_fc=tf.Variable(tf.truncated_normal([flatten_shape,fc_shape], stddev=0.1))
	weights_bias=tf.Variable(tf.constant(0.1, shape=[fc_shape]))
	fc = tf.matmul(input,weights_fc) + weights_bias
	# 没有用激励demo里面
	fc = tf.nn.relu6(fc)
	return fc

def inference_op(input, keep_prob):
	# with tf.variable_scope('conv1'):
	conv1_1 = conv_op(input, weights['conv1_1'],biases['conv1_1'])
	# conv1_2=conv_op(conv1_1,64)
	pool1 = max_pool(conv1_1)

	# with tf.variable_scope('conv2'):
	conv2_1 = conv_op(pool1, weights['conv2_1'],biases['conv2_1'])
	# conv2_2=conv_op(conv2_1,128)
	pool2 = max_pool(conv2_1)

	# with tf.variable_scope('conv3'):
	conv3_1 = conv_op(pool2, weights['conv3_1'],biases['conv3_1'])
	conv3_2 = conv_op(conv3_1, weights['conv3_2'],biases['conv3_2'])
	pool3 = max_pool(conv3_2)

	# with tf.variable_scope('conv4'):
	conv4_1 = conv_op(pool3, weights['conv4_1'],biases['conv4_1'])
	conv4_2 = conv_op(conv4_1, weights['conv4_2'],biases['conv4_2'])
	pool4 = max_pool(conv4_2)


	conv5_1 = conv_op(pool4, weights['conv5_1'],biases['conv5_1'])
	conv5_2 = conv_op(conv5_1, weights['conv5_2'],biases['conv5_2'])
	pool5 = max_pool(conv5_2)

	shape = pool5.get_shape()

	flatten_shape = shape[1].value * shape[2].value * shape[3].value
	print('flatten_shape',shape[1],shape[2],shape[3])
	flatten = tf.reshape(pool5, shape=[-1, flatten_shape])

	with tf.variable_scope('fc1'):
		fc1 = fc(flatten, flatten_shape, 4096)
	with tf.variable_scope('fc2'):
		fc2 = fc(fc1, 4096, 4096)
	with tf.variable_scope('fc3'):
		fc3 = fc(fc2, 4096, 1000)
	with tf.variable_scope('output'):
		weights_logits = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
		bias_logits = tf.Variable(tf.zeros([10]))
		logits = tf.matmul(fc3, weights_logits) + bias_logits
	return logits

def train():
	with my_graph.as_default():
		#得到预测的label
		logits=inference_op(x,keep_prob=1)
		#计算损失函数
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
			labels=y,logits=logits
			))
		#构建优化器
		train_opt=tf.train.AdamOptimizer(initial_learning_rate).minimize(loss)

		#计算模型准确率
		correct_pred=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

		import pandas as pd
		# y_train1 = one_hot_encode(y_train)
		y_train1 = pd.get_dummies(y_train)
		print(y_train1.shape)
		# y_test1 = one_hot_encode(y_test)
		y_test1 = pd.get_dummies(y_test)
		print(y_test1.shape)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			epochs=10
			for e in range(epochs):
				for i in range(1):
					# train_dict = {x: x_train,y: y_train1}
					train_dict = {x: x_train[:100], y: y_train1[:100]}
					_, train_loss = sess.run([train_opt, loss], train_dict)
					train_acc=sess.run(accuracy,train_dict)
					test_dict = {x: x_test[:100], y: y_test1[:100]}
					test_acc = sess.run(accuracy, test_dict)
					print('Epochs:{} - Train Loss:{:.5f} - Train Acc:{:.4f} - Test Acc:{:.4f}'.format(e, train_loss, train_acc, test_acc))

			# 测试集
			test_dict = {x: x_test[:100], y: y_test1[:100]}
			test_acc = sess.run(accuracy, test_dict)
			print('Test Acc', test_acc)

if __name__ == '__main__':
	train()








