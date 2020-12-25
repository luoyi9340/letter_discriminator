# -*- coding: utf-8 -*-  
'''
Created on 2020年12月23日



@author: irenebritney
'''
from model.abstract_model import AModel
from model.resnet.part import BasicBlock, Bottleneck

import tensorflow as tf


#    resnet_18网络结构
class ResNet_18(AModel):
    '''ResNet 18
        输入：100 * 100 * 1
        -----------------layer 1-------------------
        Conv:
            kernel_size=[3*3] stride=2 padding=1 active=relu norm=bn
            out=[50 * 50 * 64]
        -----------------BasicBlock 1*2-------------------
        Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[50 * 50 * 64]
        Conv:[3*3*64] stride=1 padding=1 norm=bn out=[50 * 50 * 64]   
        shortcut: out=[50 * 50 * 64]   
        active: relu
        times: 2（该层重复2次）
        -----------------BasicBlock 2-------------------
        Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[25 * 25 *128]
        Conv:[3*3*128] stride=1 padding=1 norm=bn out=[25 * 25 * 128]   
        shortcut: out=[25 * 25 * 128]   
        active: relu
        -----------------BasicBlock 2*1-------------------
        Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[25 * 25 *128]
        Conv:[3*3*128] stride=1 padding=1 norm=bn out=[25 * 25 * 128]   
        shortcut: out=[25 * 25 * 128]   
        active: relu
        times: 1（该层重复1次）
        -----------------BasicBlock 3-------------------
        Conv:[3*3*256] stride=2 padding=1 active=relu norm=bn out=[13 * 13 *256]
        Conv:[3*3*256] stride=1 padding=1 norm=bn out=[13 * 13 * 256]   
        shortcut: out=[13 * 13 * 256]   
        active: relu
        -----------------BasicBlock 3*1-------------------
        Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[13 * 13 *256]
        Conv:[3*3*256] stride=1 padding=1 norm=bn out=[13 * 13 * 256]   
        shortcut: out=[13 * 13 * 256]   
        active: relu
        times: 1（该层重复1次）
        -----------------BasicBlock 4-------------------
        Conv:[3*3*512] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 512]
        Conv:[3*3*512] stride=1 padding=1 norm=bn out=[7 * 7 * 512]   
        shortcut: out=[7 * 7 * 512]   
        active: relu
        -----------------BasicBlock 4*1-------------------
        Conv:[3*3*512] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 512]
        Conv:[3*3*512] stride=1 padding=1 norm=bn out=[7 * 7 * 512]   
        shortcut: out=[7 * 7 * 512]   
        active: relu
        times: 1（该层重复1次）
        -----------------layer 2-------------------
        Global AvgPooling: out=[1*1*512]
        FC: w=[512 * 52] active=Softmax
    '''
    def __init__(self, learning_rate=0.9):
        super(ResNet_18, self).__init__(name='ResNet_18', learning_rate=learning_rate)
        pass
    
    #    子类必须指明梯度更新方式
    def optimizer(self, net, learning_rate=0.9):
        return tf.optimizers.Adam(learning_rate=learning_rate)

    #    子类必须指明损失函数
    def loss(self):
        return tf.losses.categorical_crossentropy

    #    子类必须指明评价方式
    def metrics(self):
        return [tf.metrics.CategoricalAccuracy()]

    #    装配模型
    def assembling(self, net):
        kernel_initializer = 'uniform'
        
        #    layer1
        net.add(tf.keras.models.Sequential([
                #    坑，ZeroPadding2D放在开头第一个layer时，load_weights时会失败。。。
#                 tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.Conv2D(name='Conv_1', filters=16, kernel_size=(3, 3), strides=2, padding='valid', input_shape=(100, 100, 1), kernel_initializer=kernel_initializer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ], name='layer_1'))
        #    BasicBlock 1
        net.add(BasicBlock(name='BasicBlock_1_0', filters=[16, 16], strides=1, input_shape=(49, 49, 16), output_shape=(49, 49, 16), kernel_initializer=kernel_initializer))
        net.add(BasicBlock(name='BasicBlock_1_1', filters=[16, 16], strides=1, input_shape=(49, 49, 16), output_shape=(49, 49, 16), kernel_initializer=kernel_initializer))
        #    BasicBlock 2
        net.add(BasicBlock(name='BasicBlock_2_0', filters=[32, 32], strides=2, input_shape=(49, 49, 64), output_shape=(25, 25, 32), kernel_initializer=kernel_initializer))
        net.add(BasicBlock(name='BasicBlock_2_1', filters=[32, 32], strides=1, input_shape=(25, 25, 32), output_shape=(25, 25, 32), kernel_initializer=kernel_initializer))
        #    BasicBlock 3
        net.add(BasicBlock(name='BasicBlock_3_0', filters=[64, 64], strides=2, input_shape=(25, 25, 32), output_shape=(13, 13, 64), kernel_initializer=kernel_initializer))
        net.add(BasicBlock(name='BasicBlock_3_1', filters=[64, 64], strides=1, input_shape=(13, 13, 64), output_shape=(13, 13, 64), kernel_initializer=kernel_initializer))
        #    BasicBlock 4
        net.add(BasicBlock(name='BasicBlock_4_0', filters=[128, 128], strides=2, input_shape=(13, 13, 64), output_shape=(7, 7, 128), kernel_initializer=kernel_initializer))
        net.add(BasicBlock(name='BasicBlock_4_1', filters=[128, 128], strides=1, input_shape=(7, 7, 128), output_shape=(7, 7, 128), kernel_initializer=kernel_initializer))
        #    layer2
        net.add(tf.keras.models.Sequential([
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(46, activation='softmax')
            ], name='layer_2'))
        pass
    pass




#    ResNet_50
class ResNet_50(AModel):
    '''ResNet50
        输入：100 * 100 * 1
        -----------------layer 1-------------------
        Conv:
            kernel_size=[3*3] stride=2 padding=0 active=relu norm=bn
            out=[49 * 49 * 32]
        -----------------Bottleneck 1*3-------------------
        Conv:[1*1*32] stride=1 padding=0 active=relu norm=bn out=[49 * 49 * 32]
        Conv:[3*3*32] stride=1 padding=1 active=relu norm=bn out=[49 * 49 * 32]
        Conv:[1*1*128] stride=1 padding=0 norm=bn out=[49 * 49 * 128]
        shortcut: out=[49 * 49 * 128]   
        active: relu
        times: 3（该层重复3次）
        -----------------Bottleneck 2-------------------
        Conv:[1*1*64] stride=1 padding=0 active=relu norm=bn out=[49 * 49 * 64]
        Conv:[3*3*64] stride=2 padding=1 active=relu norm=bn out=[25 * 25 * 64]
        Conv:[1*1*256] stride=1 padding=0 norm=bn out=[25 * 25 * 128]
        shortcut: out=[25 * 25 * 256]   
        active: relu
        -----------------Bottleneck 2*3-------------------
        Conv:[1*1*64] stride=1 padding=0 active=relu norm=bn out=[25 * 25 * 64]
        Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[25 * 25 * 64]
        Conv:[1*1*256] stride=1 padding=0 norm=bn out=[25 * 25 * 128]
        shortcut: out=[25 * 25 * 256]   
        active: relu
        times: 3（该层重复3次）
        -----------------Bottleneck 3-------------------
        Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[25 * 25 * 128]
        Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[13 * 13 * 128]
        Conv:[1*1*512] stride=1 padding=0 norm=bn out=[13 * 13 * 512]
        shortcut: out=[13 * 13 * 512]   
        active: relu
        -----------------Bottleneck 3*5-------------------
        Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[13 * 13 * 128]
        Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[13 * 13 * 128]
        Conv:[1*1*512] stride=1 padding=0 norm=bn out=[13 * 13 * 256]
        shortcut: out=[13 * 13 * 512]   
        active: relu
        times: 5（该层重复5次）
        -----------------Bottleneck 4-------------------
        Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[13 * 13 * 128]
        Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 128]
        Conv:[1*1*512] stride=1 padding=0 norm=bn out=[7 * 7 * 512]
        shortcut: out=[7 * 7 * 512]   
        active: relu
        -----------------Bottleneck 4*2-------------------
        Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[7 * 7 * 256]
        Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 256]
        Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[7 * 7 * 1024]
        shortcut: out=[7 * 7 * 1024]   
        active: relu
        times: 2（该层重复2次）
        -----------------layer 2-------------------
        Global AvgPooling: out=[1*1*1024]
        FC: w=[1024 * 46] active=Softmax
    '''
    def __init__(self, learning_rate=0.9):
        super(ResNet_50, self).__init__(name='ResNet_50', learning_rate=learning_rate)
        pass
    
    #    子类必须指明梯度更新方式
    def optimizer(self, net, learning_rate=0.9):
        return tf.optimizers.Adam(learning_rate=learning_rate)

    #    子类必须指明损失函数
    def loss(self):
        return tf.losses.categorical_crossentropy

    #    子类必须指明评价方式
    def metrics(self):
        return [tf.metrics.CategoricalAccuracy()]

    #    装配模型
    def assembling(self, net):
        kernel_initializer = 'uniform'
        
        #    layer1
        net.add(tf.keras.models.Sequential([
                #    坑，ZeroPadding2D放在开头第一个layer时，load_weights时会失败。。。
                tf.keras.layers.Conv2D(name='Conv_1', filters=32, kernel_size=(3, 3), strides=2, padding='valid', input_shape=(100, 100, 1), kernel_initializer=kernel_initializer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ], name='layer_1'))
        #    Bottleneck 1
        net.add(Bottleneck(name='Bottleneck_1_0', filters=[32, 32, 128], strides=1, input_shape=(49, 49, 32), output_shape=(49, 49, 128), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_1_1', filters=[32, 32, 128], strides=1, input_shape=(49, 49, 128), output_shape=(49, 49, 128), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_1_2', filters=[32, 32, 128], strides=1, input_shape=(49, 49, 128), output_shape=(49, 49, 128), kernel_initializer=kernel_initializer))
        #    Bottleneck 2
        net.add(Bottleneck(name='Bottleneck_2_0', filters=[64, 64, 256], strides=2, input_shape=(49, 49, 128), output_shape=(25, 25, 256), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_2_1', filters=[64, 64, 256], strides=1, input_shape=(25, 25, 256), output_shape=(25, 25, 256), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_2_2', filters=[64, 64, 256], strides=1, input_shape=(25, 25, 256), output_shape=(25, 25, 256), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_2_3', filters=[64, 64, 256], strides=1, input_shape=(25, 25, 256), output_shape=(25, 25, 256), kernel_initializer=kernel_initializer))
        #    Bottleneck 3
        net.add(Bottleneck(name='Bottleneck_3_0', filters=[128, 128, 512], strides=2, input_shape=(25, 25, 256), output_shape=(13, 13, 512), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_3_1', filters=[128, 128, 512], strides=1, input_shape=(13, 13, 512), output_shape=(13, 13, 512), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_3_2', filters=[128, 128, 512], strides=1, input_shape=(13, 13, 512), output_shape=(13, 13, 512), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_3_3', filters=[128, 128, 512], strides=1, input_shape=(13, 13, 512), output_shape=(13, 13, 512), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_3_4', filters=[128, 128, 512], strides=1, input_shape=(13, 13, 512), output_shape=(13, 13, 512), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_3_5', filters=[128, 128, 512], strides=1, input_shape=(13, 13, 512), output_shape=(13, 13, 512), kernel_initializer=kernel_initializer))
        #    Bottleneck 4
        net.add(Bottleneck(name='Bottleneck_4_0', filters=[256, 256, 1024], strides=2, input_shape=(13, 13, 512), output_shape=(7, 7, 1024), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_4_1', filters=[256, 256, 1024], strides=1, input_shape=(7, 7, 1024), output_shape=(7, 7, 1024), kernel_initializer=kernel_initializer))
        net.add(Bottleneck(name='Bottleneck_4_2', filters=[256, 256, 1024], strides=1, input_shape=(7, 7, 1024), output_shape=(7, 7, 1024), kernel_initializer=kernel_initializer))
        #    layer2
        net.add(tf.keras.models.Sequential([
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(46, activation='softmax')
            ], name='layer_2'))
        pass
    pass
