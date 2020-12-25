'''
Created on 2020年12月22日

@author: irenebritney
'''
import tensorflow as tf

from model.abstract_model import AModel
from model.googlelenet.v1 import part


#    GoogleLeNetV1版本
class GoogleLeNet_V1(AModel):
    def __init__(self, learning_rate=0.9, name="GoogleLeNet_V1"):
        super(GoogleLeNet_V1, self).__init__(learning_rate, name)
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
        net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(100, 100, 1), strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(100, 100, 32)))
        #    layer 2
        net.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), input_shape=(50, 50, 32), strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(50, 50, 96)))
        #    Inception 3a
        net.add(part.Inception_V1_3a(input_shape=(25, 25, 96), kernel_initializer='uniform'))
        #    Inception 3b
        net.add(part.Inception_V1_3b(input_shape=(25, 25, 128), kernel_initializer='uniform'))
        #    layer 3
        net.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=(25, 25, 240)))
        #    Inception 4a
        net.add(part.Inception_V1_4a(input_shape=(12, 12, 240), kernel_initializer='uniform'))
        #    Inception 4b
        net.add(part.Inception_V1_4b(input_shape=(12, 12, 256), kernel_initializer='uniform'))
        #    Inception 4c
        net.add(part.Inception_V1_4c(input_shape=(12, 12, 256), kernel_initializer='uniform'))
        #    Inception 4d
        net.add(part.Inception_V1_4d(input_shape=(12, 12, 256), kernel_initializer='uniform'))
        #    Inception 4e
        net.add(part.Inception_V1_4e(input_shape=(12, 12, 264), kernel_initializer='uniform'))
        #    layer 4
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(12, 12, 416)))
        #    Inception 5a
        net.add(part.Inception_V1_5a(input_shape=(6, 6, 416), kernel_initializer='uniform'))
        #    Inception 5b
        net.add(part.Inception_V1_5b(input_shape=(6, 6, 416), kernel_initializer='uniform'))
        #    layer 5
        net.add(tf.keras.layers.AvgPool2D(pool_size=(6, 6), strides=1, padding='valid'))
        net.add(tf.keras.layers.Flatten())
        net.add(tf.keras.layers.Dense(52, activation='softmax'))
        pass
    pass
