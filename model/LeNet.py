# -*- coding: utf-8 -*-  
'''
LeNet网络


LeNet-5网络（原生）：
    输入：32 * 32 * 1（32*32的灰度图像）
    ------------------------------------
    Conv1：
        kernel_size=5*5*6 stride=1 padding=0        
        输出：28*28*6
    avg pooling：
        kernel_size=2*2 stride=2 padding=0    
        输出：14*14*6
    active：
        sigmoid
    ------------------------------------
    Conv2：
        kernel_size=5*5*16 stride=1 padding=0
        输出：10*10*16
    avg pooling:
        kernel_size=2*2 stride=2 padding=0
        输出：5*5*16
    active：
        sigmoid
    ------------------------------------
    Conv3：
        kernel_size=5*5*120 stride=1 padding=0
        输出：1*1*120
    ------------------------------------
    FC1：
        参数矩阵：[120*84]
        输出：84维向量
    sigmoid：激活
    FC2：
        参数矩阵：[84*10]
        输出：10维向量
    ------------------------------------
    Loss：
        Softmax 输出10个分类的概率
        

以原生为参考，这里的LeNet-5网络：
    输出：100 * 100 * 1（100*100的灰度图像）
    -----------------layer 1-------------------
    Conv1：
        kernel_size=5*5*8 stride=1 padding=0
        输出：96*96*8
    avg pooling:
        kernel_size=2*2 stride=2 padding=0
        输出：48*48*8
    active：
        sigmoid
    -----------------layer 2-------------------
    Conv2：
        kernel_size=5*5*32 stride=1 padding=0
        输出：44*44*32
    avg pooling：
        kernel_size=2*2 stride=2 padding=0
        输出：22*22*32
    active：
        sigmoid
    ------------------layer 3------------------
    Conv3：
        kernel_size=5*5*128 stride=1 padding=0
        输出：18*18*128
    avg pooling：
        kernel_size=2*2 stride=2 padding=0
        输出：9*9*128
    active：
        sigmoid
    ------------------layer 4------------------
    Conv4：
        kernel_size=5*5*512 stride=1 padding=0
        输出：5*5*512
    active：
        sigmoid
    Conv5：
        kernel_size=5*5*1024 stride=1 padding=0
        输出：1*1*1024
    ------------------layer 5------------------
    FC1：
        参数矩阵：[1024*512]
        输出：512维向量
    sigmoid：激活
    FC2：
        参数矩阵：[512*52]
        输出：52维向量
    ------------------layer 6------------------
    Loss：
        Softmax 输出52个分类的概率


Created on 2020年12月15日

@author: irenebritney
'''
import tensorflow as tf
from model.abstract_model import AModel


#    处理100*100的LeNet5网络
class LeNet_5(AModel):
    def __init__(self, learning_rate=0.9):
        super(LeNet_5, self).__init__(learning_rate=learning_rate, name="LeNet-5")
        pass
    
    #    子类必须指明梯度更新方式
    def optimizer(self, net, learning_rate=0.9):
        '''随机梯度下降'''
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #    子类必须指明损失函数
    def loss(self):
        '''交叉熵损失'''
        return tf.keras.losses.categorical_crossentropy
    #    子类必须指明评价方式
    def metrics(self):
        '''准确率评价'''
        return [tf.keras.metrics.Accuracy()]
    
    #    装配模型
    def assembling(self, net):
        #    layer 1
        net.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5,5), strides=1, padding='valid', input_shape=(100, 100, 1)))
        net.add(tf.keras.layers.AvgPool2D(pool_size=(2,2), strides=2, padding='valid', input_shape=(96, 96, 8)))
        net.add(tf.keras.layers.Activation("sigmoid"))
        #    layer 2
        net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=1, padding='valid', input_shape=(48, 48, 8)))
        net.add(tf.keras.layers.AvgPool2D(pool_size=(2,2), strides=2, padding='valid', input_shape=(44, 44, 32)))
        net.add(tf.keras.layers.Activation("sigmoid"))
        #    layer 3
        net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=1, padding='valid', input_shape=(22, 22, 32)))
        net.add(tf.keras.layers.AvgPool2D(pool_size=(2,2), strides=2, padding='valid', input_shape=(18, 18, 128)))
        net.add(tf.keras.layers.Activation("sigmoid"))
        #    layer 4
        net.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(5,5), strides=1, padding='valid', input_shape=(9, 9, 128)))
        net.add(tf.keras.layers.Activation("sigmoid"))
        net.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(5,5), strides=1, padding='valid', input_shape=(5, 5, 512)))
        net.add(tf.keras.layers.Activation("sigmoid"))
        net.add(tf.keras.layers.Flatten())                           #    将1*1*1024的特征图拉直为1024维向量
        #    layer 5
        net.add(tf.keras.layers.Dense(512))
        net.add(tf.keras.layers.Activation("sigmoid"))
        net.add(tf.keras.layers.Dense(52))
        net.add(tf.keras.layers.Activation("softmax"))
        pass
    
    
    pass

