# -*- coding: utf-8 -*-  
'''
Created on 2020年12月22日

@author: irenebritney
'''
import abc

import tensorflow as tf


#    Conv + BN + ReLU简化操作
class Conv2D_BN_ReLU(tf.keras.Model):
    '''Conv + BN + ReLU简化'''
    def __init__(self, name=None, filters=None, kernel_size=None, strides=None, padding=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Conv2D_BN_ReLU, self).__init__(name=name, **kwargs)
        
        self.__output_shape = output_shape
        
        self.__layer = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, input_shape=input_shape, kernel_initializer=kernel_initializer) if input_shape else tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ], name=name)
        pass
    def call(self, x, training=None, mask=None):
        y = self.__layer(x, training=training, mask=mask)
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + "outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        return y
    pass


#    Inception_V2_3x系列
class Inception_V2_3x(tf.keras.Model, metaclass=abc.ABCMeta):
    '''Inception_V2_3x系列'''
    def __init__(self, name=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_3x, self).__init__(name=name, **kwargs)
        
        #    定义输出shape
        self.__output_shape = output_shape
        
        #    装配Inception层
        conv_filters = self.conv_filters()
        #    分支1
        self.__branch1 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[0][0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer)
            ], name=name + "_branch1")
        #    分支2
        self.__branch2 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[1][0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[1][1], kernel_size=(5, 5), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name=name + "_branch2")
        #    分支3
        self.__branch3 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[2][0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][1], kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][2], kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name=name + "_branch3")
        #    分支4
        self.__branch4 = tf.keras.models.Sequential([
                tf.keras.layers.AvgPool2D(pool_size=(3, 3), strides=1, padding='same', input_shape=input_shape),
                Conv2D_BN_ReLU(filters=conv_filters[3][0], kernel_size=(1, 1), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name=name + "_branch4")
        
        pass
    #    前向
    def call(self, x, training=None, mask=None):
        '''前向传播
            将4个分支在通道层合并
        '''
        #    分别计算4个分支的前向
        y_branch1 = self.__branch1(x, training=training, mask=mask)
        y_branch2 = self.__branch2(x, training=training, mask=mask)
        y_branch3 = self.__branch3(x, training=training, mask=mask)
        y_branch4 = self.__branch4(x, training=training, mask=mask)
        
        #    4个分支的前向结果拼接
        y = tf.concat([y_branch1, y_branch2, y_branch3, y_branch4], axis=3)
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + "outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return y
    #    各层卷积核层数定义
    @abc.abstractclassmethod
    def conv_filters(self):
        pass
    pass


#    Inception_V2_3a
class Inception_V2_3a(Inception_V2_3x):
    '''Inception_V2_3a
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_3a, self).__init__(name="Inception_V2_3a", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [64],
                [48, 64],
                [64, 96, 96],
                [32]
            ]
    pass


#    Inception_V2_3b
class Inception_V2_3b(Inception_V2_3x):
    '''Inception_V2_3b
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_3b, self).__init__(name="Inception_V2_3b", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [64],
                [48, 64],
                [64, 96, 96],
                [64]
            ]
    pass


#    Inception_V2_3c
class Inception_V2_3c(Inception_V2_3x):
    '''Inception_V2_3c
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_3c, self).__init__(name="Inception_V2_3c", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [64],
                [48, 64],
                [64, 96, 96],
                [64]
            ]
    pass


#    Inception_V2_4x系列
class Inception_V2_4x(tf.keras.Model, metaclass=abc.ABCMeta):
    '''Inception_V2_4x系列
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
    '''
    def __init__(self, name=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_4x, self).__init__(name=name, **kwargs)
        
        #    定义输出shape
        self.__output_shape = output_shape
        
        #    装配Inception层
        conv_filters = self.conv_filters()
        #    分支1
        self.__branch1 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[0][0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer)
            ], name=name + "_branch1")
        #    分支2
        self.__branch2 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[1][0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[1][1], kernel_size=(1, 5), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[1][2], kernel_size=(5, 1), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name=name + "_branch2")
        #    分支3
        self.__branch3 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[2][0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][1], kernel_size=(5, 1), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][2], kernel_size=(1, 5), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][3], kernel_size=(5, 1), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][4], kernel_size=(1, 5), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name=name + "_branch3")
        #    分支4
        self.__branch4 = tf.keras.models.Sequential([
                tf.keras.layers.AvgPool2D(pool_size=(3, 3), strides=1, padding='same', input_shape=input_shape),
                Conv2D_BN_ReLU(filters=conv_filters[3][0], kernel_size=(1, 1), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name=name + "_branch4")
        
        pass
    #    前向
    def call(self, x, training=None, mask=None):
        '''前向传播
            将4个分支在通道层合并
        '''
        #    分别计算4个分支的前向
        y_branch1 = self.__branch1(x, training=training, mask=mask)
        y_branch2 = self.__branch2(x, training=training, mask=mask)
        y_branch3 = self.__branch3(x, training=training, mask=mask)
        y_branch4 = self.__branch4(x, training=training, mask=mask)
        
        #    4个分支的前向结果拼接
        y = tf.concat([y_branch1, y_branch2, y_branch3, y_branch4], axis=3)
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception("outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return y
    #    各层卷积核层数定义
    @abc.abstractclassmethod
    def conv_filters(self):
        pass
    pass
#    Inception_V2_4a
class Inception_V2_4a(tf.keras.Model):
    '''Inception_V2_4a
        分为3支：
            分支1:
                Conv:[3*3*384] stride=2 padding=0 norm=BN active=ReLU out=[10*10*384]
            分支2:
                Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
                Conv:[3*3*96] stride=2 padding=0 norm=BN active=ReLU out=[10*10*96]
                Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[10*10*96]
            分支3:
                max pooling:[3*3] stride=2 padding=0 out=[10*10*288]
            out=[10*10*(384+96+288)]=[10*10*768]
    '''
    def __init__(self, name=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_4a, self).__init__(name='Inception_V2_4a', **kwargs)
        
        self.__output_shape = output_shape
        
        #    拼装layer
        #    分支1
        self.__branch1 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=384, kernel_size=(3, 3), strides=2, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer)
            ], name='Inception_V2_4a_branch1')
        #    分支2
        self.__branch2 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=64, kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=96, kernel_size=(3, 3), strides=2, padding='valid', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=96, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=kernel_initializer)
            ], name='Inception_V2_4a_branch2')
        #    分支3
        self.__branch3 = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=input_shape)
            ], name='Inception_V2_4a_branch3')
#         pass
    #    前向传播
    def call(self, x, training=None, mask=None):
        y_branch1 = self.__branch1(x, training=training, mask=mask)
        y_branch2 = self.__branch2(x, training=training, mask=mask)
        y_branch3 = self.__branch3(x, training=training, mask=mask)
        
        #    拼接3个分支的输出
        y = tf.concat([y_branch1, y_branch2, y_branch3], axis=3)
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception("Inception:" + self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
 
        return y
    pass

#    Inception_V2_4b
class Inception_V2_4b(Inception_V2_4x):
    '''Inception_V2_4b
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_4b, self).__init__(name="Inception_V2_4b", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [192],
                [128, 128, 192],
                [128, 128, 128, 128, 192],
                [192]
            ]
    pass

#    Inception_V2_4c
class Inception_V2_4c(Inception_V2_4x):
    '''Inception_V2_4c
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_4c, self).__init__(name="Inception_V2_4c", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [192],
                [160, 160, 192],
                [160, 160, 160, 160, 192],
                [192]
            ]
    pass

#    Inception_V2_4d
class Inception_V2_4d(Inception_V2_4x):
    '''Inception_V2_4d
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_4d, self).__init__(name="Inception_V2_4d", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [192],
                [160, 160, 192],
                [160, 160, 160, 160, 192],
                [192]
            ]
    pass

#    Inception_V2_4e
class Inception_V2_4e(Inception_V2_4x):
    '''Inception_V2_4e
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_4e, self).__init__(name="Inception_V2_4e", input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [192],
                [192, 192, 192],
                [192, 192, 192, 192, 192],
                [192]
            ]
    pass



#    Inception_V2_5x_branch
class Inception_V2_5x_branch(tf.keras.Model):
    '''Inception_V2_5x_branch
        分为2支：
            分支1：Conv:[1*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
            分支2：Conv:[3*1*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
            out=[4*4*(192+192)]=[4*4*384]
    '''
    def __init__(self, filters=None, name=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_5x_branch, self).__init__(name=name, **kwargs)
        
        self.__output_shape = output_shape
        
        #    第1个卷积分支
        self.__branch_conv1 = Conv2D_BN_ReLU(filters=filters[0], kernel_size=(1, 3), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer)
        
        #    第2个卷积分支
        self.__branch_conv2 = Conv2D_BN_ReLU(filters=filters[1], kernel_size=(3, 1), strides=1, padding='same', input_shape=input_shape, kernel_initializer=kernel_initializer)
        pass
    def call(self, x, training=None, mask=None):
        y_branch1 = self.__branch_conv1(x, training=training, mask=mask)
        y_branch2 = self.__branch_conv2(x, training=training, mask=mask)
        
        y = tf.concat([y_branch1, y_branch2], axis=3)
        
        #    如果设置了output_shape则做检测
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception("Inception:" + self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
            
        return y
    pass
#    Inception_V2_5x系列
class Inception_V2_5x(tf.keras.Model, metaclass=abc.ABCMeta):
    '''Inception_V2_5x
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
                Conv:[1*1*96] stride=1 padding=0 norm=BN active=ReLU out=[4*4*96]
            out=[4*4*(160+384+384+96)]=[4*4*1024]
    '''
    def __init__(self, name=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_5x, self).__init__(name=name, **kwargs)
        
        self.__output_shape = output_shape
        
        #    拼装layer
        conv_filters = self.conv_filters()
        #    分支1
        self.__branch1 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[0][0], kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer)
            ], name=name + '_branch1')
        #    分支2
        self.__branch2 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[1][0], kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Inception_V2_5x_branch(filters=[conv_filters[1][1], conv_filters[1][2]])
            ], name=name + '_branch2')
        #    分支3
        self.__branch3 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=conv_filters[2][0], kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=conv_filters[2][1], kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Inception_V2_5x_branch(filters=[conv_filters[2][2], conv_filters[2][3]])
            ], name=name + '_branch3')
        #    分支4
        self.__branch4 = tf.keras.models.Sequential([
                tf.keras.layers.AvgPool2D(pool_size=(3, 3), strides=1, padding='same', input_shape=input_shape),
                Conv2D_BN_ReLU(filters=conv_filters[3][0], kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer=kernel_initializer)
            ], name=name + '_branch4')
        pass
    #    前向传播
    def call(self, x, training=None, mask=None):
        y_branch1 = self.__branch1(x, training=training, mask=mask)
        y_branch2 = self.__branch2(x, training=training, mask=mask)
        y_branch3 = self.__branch3(x, training=training, mask=mask)
        y_branch4 = self.__branch4(x, training=training, mask=mask)
        
        #    拼接3个分支的输出
        y = tf.concat([y_branch1, y_branch2, y_branch3, y_branch4], axis=3)
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception("Inception:" + self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
 
        return y
    
    @abc.abstractclassmethod
    def conv_filters(self):
        pass
    pass

#    Inception_V2_5a
class Inception_V2_5a(tf.keras.Model):
    '''Inception_V2_5a
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_5a, self).__init__(name="Inception_V2_5a", **kwargs)
        
        self.__output_shape = output_shape
        
        #    拼装layer
        #    分支1
        self.__branch1 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=192, kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=320, kernel_size=(3, 3), strides=2, padding='valid', kernel_initializer=kernel_initializer)
            ], name='Inception_V2_5a_branch1')
        #    分支2
        self.__branch2 = tf.keras.models.Sequential([
                Conv2D_BN_ReLU(filters=192, kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=192, kernel_size=(1, 5), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=192, kernel_size=(5, 1), strides=1, padding='same', kernel_initializer=kernel_initializer),
                Conv2D_BN_ReLU(filters=192, kernel_size=(3, 3), strides=2, padding='valid', kernel_initializer=kernel_initializer)
            ], name='Inception_V2_5a_branch2')
        #    分支3
        self.__branch3 = tf.keras.models.Sequential([
                tf.keras.layers.AvgPool2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=input_shape)
            ], name='Inception_V2_5a_branch3')
        pass
    #    前向传播
    def call(self, x, training=None, mask=None):
        y_branch1 = self.__branch1(x, training=training, mask=mask)
        y_branch2 = self.__branch2(x, training=training, mask=mask)
        y_branch3 = self.__branch3(x, training=training, mask=mask)
        
        #    拼接3个分支的输出
        y = tf.concat([y_branch1, y_branch2, y_branch3], axis=3)
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception("Inception:" + self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
 
        return y    
    pass
    
#    Inception_V2_5b
class Inception_V2_5b(Inception_V2_5x):
    '''Inception_V2_5b
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
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_5b, self).__init__(name='Inception_V2_5b', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [160],
                [192, 192, 192],
                [224, 192, 192, 192],
                [96]
            ]
    pass

#    Inception_V2_5c
class Inception_V2_5c(Inception_V2_5x):
    '''Inception_V2_5c
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
                    分支1：Conv:[1*3*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                    分支2：Conv:[3*1*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                    out=[4*4*(96+96)]=[4*4*192]
            分支4：    
                avg pooling:[3*3] stride=1 padding=1 out=[4*4*2048]
                Conv:[1*1*48] stride=1 padding=1 norm=BN active=ReLU out=[4*4*48]
            out=[4*4*(80+192+192+48)]=[4*4*512]
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V2_5c, self).__init__(name='Inception_V2_5c', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                [80],
                [96, 96, 96],
                [112, 96, 96, 96],
                [48]
            ]
    pass
