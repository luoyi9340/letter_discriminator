# -*- coding: utf-8 -*-  
'''
DenseNet需要用到的零件

DenseBasicLayer
    Conv: 1*1
    Conv: 3*3 
    concat

DenseBlockLayer
    若干个DenseBasicLayer的组合


TransitionLayer
    Conv: 1*1
    avg_pooling: [2*2]

@author: luoyi
Created on 2021年2月20日
'''
import tensorflow as tf
from math import ceil



#    DenseLayer组件
class DenseBasicLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name='DenseBasicLayer',
                 growth_rate=12,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        '''
            @param name: layer名称
            @param growth_rate: 基本通道数
            @param input_shape: 输入格式
        '''
        super(DenseBasicLayer, self).__init__(name=name, **kwargs)
        
        self._output_shape = output_shape
        
        #    两个卷积层
        self._conv11_layer = tf.keras.models.Sequential([
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(),
                                    tf.keras.layers.Conv2D(filters=growth_rate * 4,
                                                           input_shape=input_shape,
                                                           kernel_size=[1, 1],
                                                           strides=1,
                                                           padding='same',
                                                           kernel_initializer=tf.initializers.ones(),
                                                           bias_initializer=tf.keras.initializers.zeros())
                                ], name=name + '_conv11')
        
        conv33_input_shape = (input_shape[0], input_shape[1], growth_rate * 4)
        self._conv33_layer = tf.keras.models.Sequential([
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(),
                                    tf.keras.layers.Conv2D(filters=growth_rate,
                                                           input_shape=conv33_input_shape,
                                                           kernel_size=[3, 3],
                                                           strides=1,
                                                           padding='same',
                                                           kernel_initializer=tf.initializers.he_normal(),
                                                           bias_initializer=tf.keras.initializers.zeros())
                                ], name=name + '_conv33')
        
        pass
    
    #    前向传播
    def call(self, x, **kwargs):
        y = self._conv11_layer(x)
        y = self._conv33_layer(y)
        y = tf.concat([x,y], axis=-1)
        
        #    检测输出格式是否一致
        if (self._output_shape):
            if (self._output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + " outputshape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
            pass
        
        return y
    pass


#    DenseBlockLayer
class DenseBlockLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name='DenseBlockLayer',
                 growth_rate=12,
                 num=6,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        '''
            @param name: layer名称
            @param growth_rate: 基础通道数
            @param num: 循环多少次
        '''
        super(DenseBlockLayer, self).__init__(name=name, **kwargs)
        
        self._output_shape = output_shape
        
        self._conv_layer = tf.keras.models.Sequential(name=name + '_conv')
        
        filters = input_shape[-1]
        for i in range(num):
            #    计算每层的输入输出
            layer_input_shape = (input_shape[0], input_shape[1], filters)
            filters += growth_rate
            layer_output_shape = (input_shape[0], input_shape[1], filters)
            
            self._conv_layer.add(DenseBasicLayer(name=name+'_basic_layer_' + str(i), 
                                                 growth_rate=growth_rate,
                                                 input_shape=layer_input_shape,
                                                 output_shape=layer_output_shape,
                                                 ))
            pass
        pass
    
    def call(self, x, **kwargs):
        y = self._conv_layer(x)
        
        #    检测输出格式是否一致
        if (self._output_shape):
            if (self._output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + " outputshape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
            pass
        return y
    pass


#    TransitionLayer
class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name='TransitionLayer',
                 compression_rate=0.5,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        '''
            @param name: layer名称
            @param prev_filters: 上层通道数
            @param compression_rate: 通道收缩比例(0,1]
        '''
        super(TransitionLayer, self).__init__(name=name, **kwargs)
        
        self._output_shape = output_shape
        
        self._conv11_layer = tf.keras.models.Sequential([
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.ReLU(),
                                    tf.keras.layers.Conv2D(filters=ceil(input_shape[-1] * compression_rate),
                                                           input_shape=input_shape,
                                                           kernel_size=[1,1],
                                                           strides=1,
                                                           padding='same',
                                                           kernel_initializer=tf.initializers.he_normal(),
                                                           bias_initializer=tf.keras.initializers.zeros())
                                ], name=name + '_conv11')
        
        self._pooling_layer = tf.keras.layers.AvgPool2D(pool_size=[2,2], strides=2, padding='valid')
        pass
    
    def call(self, x, **kwargs):
        y = self._conv11_layer(x)
        y = self._pooling_layer(y)
        
        #    检测输出格式是否一致
        if (self._output_shape):
            if (self._output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + " outputshape:" + str(self._output_shape) + " not equal y:" + str(y.shape))
            pass
        return y
    pass



