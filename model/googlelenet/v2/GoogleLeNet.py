# -*- coding: utf-8 -*-  
'''
GoogleLeNet V2版本（其实是V3版本）
    - 不包括辅助分类器（作者自述其实这东西对模型收敛没多大作用）


GoogleLeNet_V2简化版网络
    输入：100 * 100 * 1
    -----------------layer 1-------------------
    Conv:
        kernel_size=[3*3*32] stride=1 padding=0 norm=bn active=relu
        out=98*98*32
    Conv:
        kernel_size=[3*3*32] stride=1 padding=0 norm=bn active=relu
        out=96*96*32
    Conv:
        kernel_size=[3*3*64] stride=1 padding=1 norm=bn active=relu
        out=96*96*64
    max pooling:
        kernel_size=[2*2] stride=2 pading=0
        out=48*48*64
    -----------------layer 2-------------------
    Conv:
        kernel_size=[3*3*80] stride=1 padding=0 norm=bn active=relu 
        out=45*45*80
    Conv:
        kernel_size=[3*3*192] stride=1 padding=1 norm=bn active=relu 
        out=45*45*192
    max pooling:
        kernel_size=[3*3] stride=2 pading=0 
        out=22*22*192
    -----------------Inception 3a-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[22*22*48]
            Conv:[5*5*64] stride=1 padding=2 norm=BN active=ReLU out=[22*22*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[22*22*192]
            Conv:[1*1*32] stride=1 padding=0 out=[22*22*32]
        out=[22*22*(64+64+96+32)]=[22*22*256]
    -----------------Inception 3b-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[22*22*48]
            Conv:[5*5*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[22*22*256]
            Conv:[1*1*64] stride=1 padding=1 out=[22*22*64]
        out=[22*22*(64+64+96+64)]=[22*22*288]
    -----------------Inception 3c-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[22*22*48]
            Conv:[5*5*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[22*22*288]
            Conv:[1*1*64] stride=1 padding=1 out=[22*22*64]
        out=[22*22*(64+64+96+64)]=[22*22*288]
    -----------------Inception 4a-------------------    
    分为3支：
        分支1:
            Conv:[3*3*384] stride=2 padding=0 norm=BN active=ReLU out=[10*10*384]
        分支2:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=0 norm=BN active=ReLU out=[10*10*96]
            Conv:[3*3*96] stride=2 padding=0 norm=BN active=ReLU out=[10*10*96]
        分支3:
            max pooling:[3*3] stride=2 padding=0 out=[10*10*288]
        out=[10*10*(384+96+288)]=[10*10*768]
    -----------------Inception 4b-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[10*10*128]
            Conv:[1*5*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[10*10*128]
            Conv:[5*1*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[1*5*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[5*1*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
    -----------------Inception 4c-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
    -----------------Inception 4d-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
    -----------------Inception 4e-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
        
        out外接辅助分类器（作者后来自己也说，辅助分类器其实并没有起到加速收敛的作用。。。）：
            avg pooling:[5*5] stride=3 padding=0 out=[5*5*768]
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[5*5*128]
            Conv:[5*5*1024] stride=1 padding=0 norm=BN active=ReLU out=[1*1*1024]
            Fatten:out=[1024]
            FC:w=[1024 * 1000] active=Softmax out=[1000]
    -----------------Inception 5a-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[3*3*320] stride=2 padding=0 norm=BN active=ReLU out=[4*4*320]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[3*3*192] stride=2 padding=0 norm=BN active=ReLU out=[4*4*192]
        分支3:
            avg pooling:[3*3] stride=2 padding=0 out=[4*4*768]
        out=[4*4*(320+192+768)]=[4*4*1280]
    -----------------Inception 5b-------------------    
    分为3支：
        分支1:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[4*4*160]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[4*4*192]
            分为2支：
                分支1：Conv:[1*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                分支2：Conv:[3*1*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                out=[4*4*(192+192)]=[4*4*384]
        分支3:
            Conv:[1*1*224] stride=1 padding=0 norm=BN active=ReLU out=[4*4*224]
            Conv:[3*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
            分为2支：
                分支1：Conv:[1*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                分支2：Conv:[3*1*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                out=[4*4*(192+192)]=[4*4*384]
        分支4：    
            avg pooling:[3*3] stride=1 padding=1 out=[4*4*1280]
            Conv:[1*1*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
        out=[4*4*(160+384+384+96)]=[4*4*1024]
    -----------------Inception 5c-------------------    
    分为3支：
        分支1:
            Conv:[1*1*80] stride=1 padding=0 norm=BN active=ReLU out=[4*4*80]
        分支2:
            Conv:[1*1*96] stride=1 padding=0 norm=BN active=ReLU out=[4*4*96]
            分为2支：
                分支1：Conv:[1*3*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                分支2：Conv:[3*1*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                out=[4*4*(96+96)]=[4*4*192]
        分支3:
            Conv:[1*1*112] stride=1 padding=0 norm=BN active=ReLU out=[4*4*112]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
            分为2支：
                分支1：Conv:[1*3*384] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                分支2：Conv:[3*1*384] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                out=[4*4*(96+96)]=[4*4*192]
        分支4：    
            avg pooling:[3*3] stride=1 padding=1 out=[4*4*2048]
            Conv:[1*1*48] stride=1 padding=1 norm=BN active=ReLU out=[4*4*48]
        out=[4*4*(80+192+192+48)]=[4*4*512]
    -----------------layer 3-------------------
    avg pool:
        kernel_size=[4*4] stride=1 padding=0 
        out=[1*1*512]
    Conv:
        kernel_size=[1*1*52] stride=1 padding=0 norm=BN active=ReLU
        out=[1*1*52]
    Fatten:
        out=[52]
    Softmax:
        out=[52]个分类的概率

Created on 2020年12月22日

@author: irenebritney
'''
import tensorflow as tf

from model.abstract_model import AModel
from model.googlelenet.v2 import part



#    GoogleLeNet V2
class GoogleLeNet_V2(AModel):
    def __init__(self, learning_rate=0.9, name="GoogleLeNet_V2"):
        super(GoogleLeNet_V2, self).__init__(learning_rate, name)
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
        kernel_initializer = tf.keras.initializers.random_normal
        
        #    layer 1
        net.add(tf.keras.models.Sequential([
                part.Conv2D_BN_ReLU(filters=32, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(100, 100, 1), kernel_initializer=kernel_initializer),
                part.Conv2D_BN_ReLU(filters=32, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(98, 98, 32), kernel_initializer=kernel_initializer),
                part.Conv2D_BN_ReLU(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(96, 96, 32), kernel_initializer=kernel_initializer),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
            ], name='layer_1'))
        #    layer 2
        net.add(tf.keras.models.Sequential([
                part.Conv2D_BN_ReLU(filters=80, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(48, 48, 64), kernel_initializer=kernel_initializer),
                part.Conv2D_BN_ReLU(filters=192, kernel_size=(3, 3), strides=1, padding='same', input_shape=(45, 45, 80), kernel_initializer=kernel_initializer),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')
            ], name='layer_2'))
        #    Inception 3a -> 3c
        net.add(part.Inception_V2_3a(input_shape=(22, 22, 192), output_shape=(22, 22, 256), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_3b(input_shape=(22, 22, 256), output_shape=(22, 22, 288), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_3c(input_shape=(22, 22, 288), output_shape=(22, 22, 288), kernel_initializer=kernel_initializer))
        #    Inception 4a -> 4e
        net.add(part.Inception_V2_4a(input_shape=(22, 22, 288), output_shape=(10, 10, 768), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_4b(input_shape=(10, 10, 768), output_shape=(10, 10, 768), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_4c(input_shape=(10, 10, 768), output_shape=(10, 10, 768), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_4d(input_shape=(10, 10, 768), output_shape=(10, 10, 768), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_4e(input_shape=(10, 10, 768), output_shape=(10, 10, 768), kernel_initializer=kernel_initializer))
        #    Inception 5a -> 5c
        net.add(part.Inception_V2_5a(input_shape=(10, 10, 768), output_shape=(4, 4, 1280), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_5b(input_shape=(4, 4, 1280), output_shape=(4, 4, 1024), kernel_initializer=kernel_initializer))
        net.add(part.Inception_V2_5c(input_shape=(4, 4, 1024), output_shape=(4, 4, 512), kernel_initializer=kernel_initializer))
        #    layer 3
        net.add(tf.keras.models.Sequential([
                tf.keras.layers.AvgPool2D(pool_size=(4, 4), strides=1, padding='valid', input_shape=(4, 4, 512)),
                part.Conv2D_BN_ReLU(filters=52, kernel_size=(1, 1), strides=1, padding='valid', input_shape=(1, 1, 512), output_shape=(1, 1, 52)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Softmax()
            ], name='layer_3'))
        
        pass
    pass
