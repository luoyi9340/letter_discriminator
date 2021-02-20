# -*- coding: utf-8 -*-  
'''
ResNeXtBlock_A
    step1：分组卷积，每个分组采用完整输入数据，且直接输出完整通道数
    step2：分组卷积结果求和
    step3：step2结果与输入求和

ResNeXtBlock_B
    step1：分组卷积，每个分组采用完整输入数据，但只输出小块通道数
    step2：分组卷积结果叠加
    step3：step2结果与输入求和

ResNeXtBlock_C
    step1：整体做1*1卷积，降维
    step2：分组卷积，每个分组只负责一小块输入数据
    step3：叠加分组卷积结果
    step4：叠加后的结果过1*1卷积核，维度调整
    step4：step3结果与输入求和


@author: luoyi
Created on 2021年2月20日
'''
import tensorflow as tf


#    ResNeXtBlock_A
class ResNeXtBlock_A(tf.keras.layers.Layer):
    def __init__(self,
                 name='ResNeXtBlock_A',
                 group=32,
                 base_filters=4,
                 is_down_sample=True,
                 input_shape=None,
                 output_shape=None,
                 kernel_initializer=tf.initializers.he_normal(),
                 bias_initializer=tf.initializers.zeros(),
                 **kwargs):
        ''' 分组卷积，每组用全量数据
            @param name: 层名称
            @param group: 分组数
            @param base_filters: 每个小分组通道数
            @param is_down_sample: 是否降采样
            @param input_shape: 输入形状
            @param output_shape: 输出形状
        '''
        super(ResNeXtBlock_A, self).__init__(name=name, **kwargs)
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        #    初始化每组卷积
        self._convs = []
        for i in range(group):
            conv = tf.keras.models.Sequential(name=name + '_conv_' + str(i))
            #    1*1卷积核。降维
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv11' + str(i),
                                            filters=base_filters,
                                            kernel_size=[1, 1],
                                            strides=1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            #    3*3卷积核。降维 或 融合特征
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.ZeroPadding2D(padding=1))
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv33' + str(i),
                                            filters=base_filters,
                                            kernel_size=[3, 3],
                                            strides=2 if is_down_sample else 1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            #    1*1卷积核。升维
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv11' + str(i),
                                            filters=output_shape[-1],
                                            kernel_size=[1, 1],
                                            strides=1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            self._convs.append(conv)
            pass
        
        #    如果需要降采样（每次降采样都缩放为原图1/2），或者 通道数不对，则指定x线性变换
        if (is_down_sample or input_shape[-1] != output_shape[-1]):
            self._adjust_x = True
            self._down_sample = tf.keras.layers.Conv2D(filters=output_shape[-1],
                                                       kernel_size=[1,1],
                                                       strides=2 if is_down_sample else 1,
                                                       padding='VALID',
                                                       trainable=False,
                                                       kernel_initializer=tf.initializers.ones(),
                                                       bias_initializer=tf.initializers.zeros())
            pass
        pass
    def call(self, x, **kwargs):
        y = tf.zeros(shape=self._output_shape)
        
        #    计算每组卷积
        for conv in self._convs:
            y_ = conv(x)
            y = tf.keras.layers.add([y, y_])
            pass
        
        #    追加x
        if (self._adjust_x):
            x = self._down_sample(x)
            pass
        
        y = tf.keras.layers.add([x, y])
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self._output_shape is not None
            and self._output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass


#    ResNeXtBlock_B
class ResNeXtBlock_B(tf.keras.layers.Layer):
    def __init__(self,
                 name='ResNeXtBlock_B',
                 group=32,
                 base_filters=4,
                 is_down_sample=True,
                 input_shape=None,
                 output_shape=None,
                 kernel_initializer=tf.initializers.he_normal(),
                 bias_initializer=tf.initializers.zeros(),
                 **kwargs):
        ''' 分组卷积，每组用全量数据
            @param name: 层名称
            @param group: 分组数
            @param base_filters: 每个小分组通道数
            @param is_down_sample: 是否降采样
            @param input_shape: 输入形状
            @param output_shape: 输出形状
        '''
        super(ResNeXtBlock_B, self).__init__(name=name, **kwargs)
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        #    初始化每组卷积
        self._convs = []
        for i in range(group):
            conv = tf.keras.models.Sequential(name=name + '_conv_' + str(i))
            #    1*1卷积核。降维
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv11' + str(i),
                                            filters=base_filters,
                                            kernel_size=[1, 1],
                                            strides=1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            #    3*3卷积核。降维 或 融合特征
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.ZeroPadding2D(padding=1))
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv33' + str(i),
                                            filters=base_filters,
                                            kernel_size=[3, 3],
                                            strides=2 if is_down_sample else 1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            self._convs.append(conv)
            pass
        
        #    如果需要降采样（每次降采样都缩放为原图1/2），或者 通道数不对，则指定x线性变换
        if (is_down_sample or input_shape[-1] != output_shape[-1]):
            self._adjust_x = True
            self._down_sample = tf.keras.layers.Conv2D(filters=output_shape[-1],
                                                       kernel_size=[1,1],
                                                       strides=2 if is_down_sample else 1,
                                                       padding='VALID',
                                                       trainable=False,
                                                       kernel_initializer=tf.initializers.ones(),
                                                       bias_initializer=tf.initializers.zeros())
            pass
        pass
    def call(self, x, **kwargs):
        y = []
        #    计算每组卷积
        for conv in self._convs:
            y_ = conv(x)
            y.append(y_)
            pass
        y = tf.concat(y, axis=-1)
        
        #    追加x
        if (self._adjust_x):
            x = self._down_sample(x)
            pass
        
        y = tf.keras.layers.add([x, y])
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self._output_shape is not None
            and self._output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return y
    pass


#    ResNeXtBlock_C
class ResNeXtBlock_C(tf.keras.layers.Layer):
    def __init__(self,
                 name='ResNeXtBlock_C',
                 group=32,
                 base_filters=4,
                 is_down_sample=True,
                 input_shape=None,
                 output_shape=None,
                 kernel_initializer=tf.initializers.he_normal(),
                 bias_initializer=tf.initializers.zeros(),
                 **kwargs):
        ''' 分组卷积，每组只用一小部分数据
            @param name: 层名称
            @param group: 分组数
            @param base_filters: 每个小分组通道数
            @param is_down_sample: 是否降采样
            @param input_shape: 输入形状
            @param output_shape: 输出形状
        '''
        super(ResNeXtBlock_C, self).__init__(name=name, **kwargs)
        
        #    必要的检测，input_shape[-1]必须是base_filters的整数倍
        if (input_shape[-1] % base_filters > 0):
            raise Exception('input_shape.filters must be integral multiple then base_filters. input_shape.filters:' + str(input_shape[-1]) + ' base_filters:' + str(base_filters))
        
        self._group = group
        
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        #    初始化每组卷积
        self._convs = []
        for i in range(group):
            conv = tf.keras.models.Sequential(name=name + '_conv_' + str(i))
            #    1*1卷积核。降维
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv11' + str(i),
                                            filters=base_filters,
                                            kernel_size=[1, 1],
                                            strides=1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            #    3*3卷积核。降维 或 融合特征
            conv.add(tf.keras.layers.BatchNormalization())
            conv.add(tf.keras.layers.ReLU())
            conv.add(tf.keras.layers.ZeroPadding2D(padding=1))
            conv.add(tf.keras.layers.Conv2D(name=name + '_conv33' + str(i),
                                            filters=base_filters,
                                            kernel_size=[3, 3],
                                            strides=2 if is_down_sample else 1,
                                            padding='VALID',
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer))
            
            self._convs.append(conv)
            pass
        
        #    维度调整
        self._conv11_dims = tf.keras.models.Sequential([
                                                    tf.keras.layers.BatchNormalization(),
                                                    tf.keras.layers.ReLU(),
                                                    tf.keras.layers.Conv2D(name=name + '_conv11',
                                                                           filters=output_shape[-1],
                                                                           kernel_size=[1,1],
                                                                           strides=1,
                                                                           padding='VALID',
                                                                           kernel_initializer=kernel_initializer,
                                                                           bias_initializer=bias_initializer)
                                                  ], name=name + '_conv11_dims')
        
        #    如果需要降采样（每次降采样都缩放为原图1/2），或者 通道数不对，则指定x线性变换
        self._adjust_x = False
        if (is_down_sample or input_shape[-1] != output_shape[-1]):
            self._adjust_x = True
            self._down_sample = tf.keras.layers.Conv2D(filters=output_shape[-1],
                                                       kernel_size=[1,1],
                                                       strides=2 if is_down_sample else 1,
                                                       padding='VALID',
                                                       trainable=False,
                                                       kernel_initializer=tf.initializers.ones(),
                                                       bias_initializer=tf.initializers.zeros())
            pass
        pass
    def call(self, x, **kwargs):
        y = []
        x_split = tf.split(x, num_or_size_splits=self._group, axis=-1)
        #    计算每组卷积
        for conv, x_s in zip(self._convs, x_split):
            y_ = conv(x_s)
            y.append(y_)
            pass
        y = tf.concat(y, axis=-1)
        y = self._conv11_dims(y)
        
        #    追加x
        if (self._adjust_x):
            x = self._down_sample(x)
            pass
        
        y = tf.keras.layers.add([x, y])
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self._output_shape is not None
            and self._output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return y


