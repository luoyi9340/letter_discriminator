# -*- coding: utf-8 -*-  
'''

GoogleLeNet 零件
    Java养成的习惯，不想一个文件弄太多东西。。。
    
    - V1版本：
        - Inception_3a
        - Inception_3b
        - Inception_4a
        - Inception_4b
        - Inception_4c
        - Inception_4d
        - Inception_4e
        - Inception_5a
        - Inception_5b
        
    - V2版本：

Created on 2020年12月22日

@author: irenebritney
'''
import abc
import tensorflow as tf


#    Inception V1版本
class Inception_V1(tf.keras.Model, metaclass=abc.ABCMeta):
    def __init__(self, name=None, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1, self).__init__(name=name, **kwargs)
        
        #    定义输出shape
        self.__output_shape = output_shape
        
        #    装配Inception_3a层
        conv_filters = self.conv_filters()
        #    分支1
        self.__branch1 = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=conv_filters[0], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, activation='relu', kernel_initializer=kernel_initializer)
            ], name=name + "_branch1")
        #    分支2
        self.__branch2 = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=conv_filters[1][1], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, activation='relu', kernel_initializer=kernel_initializer),
                tf.keras.layers.Conv2D(filters=conv_filters[1][2], kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer=kernel_initializer)
            ], name=name + "_branch2")
        #    分支3
        self.__branch3 = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=conv_filters[2][1], kernel_size=(1, 1), strides=1, padding='same', input_shape=input_shape, activation='relu', kernel_initializer=kernel_initializer),
                tf.keras.layers.Conv2D(filters=conv_filters[2][2], kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer=kernel_initializer)
            ], name=name + "_branch3")
        #    分支4
        self.__branch4 = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same', input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=conv_filters[3], kernel_size=(1, 1), strides=1, padding='same', activation='relu', kernel_initializer=kernel_initializer)
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
        if (self.__output_shape
            & self.__output_shape != y.shape):
                raise Exception("Inception:" + self.name + " outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return y
    #    各层卷积核层数定义
    @abc.abstractclassmethod
    def conv_filters(self):
        pass
    pass

#    Inception_3a
class Inception_V1_3a(Inception_V1):
    '''Inception_3a
        分为4支：
            分支1:
                Conv:[1*1*32] stride=1 padding=0 active=ReLU out=[25*25*32]
            分支2:
                Conv:[1*1*48] stride=1 padding=1 active=ReLU out=[25*25*48]
                Conv:[3*3*64] stride=1 padding=1 active=ReLU out=[25*25*64]
            分支3:
                Conv:[1*1*8] stride=1 padding=1 active=ReLU out=[25*25*8]
                Conv:[5*5*16] stride=1 padding=2 active=ReLU out=[25*25*16]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[25*25*96]
                Conv:[1*1*16] stride=1 padding=1 out=[25*25*16]
        out=[25*25*(32+64+16+16)]=[25*25*128]
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_3a, self).__init__(name='Inception_V1_3a', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                32,
                [48, 64],
                [8, 16],
                16
            ]
    pass

#    Inception_3b
class Inception_V1_3b(Inception_V1):
    '''Inception_3b
        分为4支：
            分支1:
                Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[25*25*64]
            分支2:
                Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[25*25*64]
                Conv:[3*3*96] stride=1 padding=1 active=ReLU out=[25*25*96]
            分支3:
                Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[25*25*16]
                Conv:[5*5*48] stride=1 padding=2 active=ReLU out=[25*25*48]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[25*25*128]
                Conv:[1*1*32] stride=1 padding=1 out=[25*25*32]
        out=[25*25*(64+96+48+32)]=[25*25*240]
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_3b, self).__init__(name='Inception_V1_3b', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                64,
                [64, 96],
                [16, 48],
                32
            ]
    pass

#    Inception_4a
class Inception_V1_4a(Inception_V1):
    '''Inception_4a
        分为4支：
            分支1:
                Conv:[1*1*96] stride=1 padding=0 active=ReLU out=[12*12*96]
            分支2:
                Conv:[1*1*48] stride=1 padding=0 active=ReLU out=[12*12*48]
                Conv:[3*3*104] stride=1 padding=1 active=ReLU out=[14*14*104]
            分支3:
                Conv:[1*1*8] stride=1 padding=0 active=ReLU out=[12*12*8]
                Conv:[5*5*24] stride=1 padding=2 active=ReLU out=[12*12*24]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[12*12*240]
                Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(96+104+24+32)]=[12*12*256]
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_4a, self).__init__(name='Inception_V1_4a', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                96,
                [48, 104],
                [8, 24],
                32
            ]
    pass

#    Inception_4b
class Inception_V1_4b(Inception_V1):
    '''Inception_4b
        分为4支：
            分支1:
                Conv:[1*1*80] stride=1 padding=0 active=ReLU out=[12*12*80]
            分支2:
                Conv:[1*1*56] stride=1 padding=0 active=ReLU out=[12*12*56]
                Conv:[3*3*112] stride=1 padding=1 active=ReLU out=[12*12*112]
            分支3:
                Conv:[1*1*12] stride=1 padding=0 active=ReLU out=[12*12*12]
                Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[12*12*32]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[12*12*256]
                Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(80+112+32+32)]=[12*12*256]  
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_4b, self).__init__(name='Inception_V1_4b', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                80,
                [56, 112],
                [12, 32],
                32
            ]
    pass

#    Inception_4c
class Inception_V1_4c(Inception_V1):
    '''Inception_4c
        分为4支：
            分支1:
                Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[12*12*64]
            分支2:
                Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[12*12*64]
                Conv:[3*3*128] stride=1 padding=1 active=ReLU out=[12*12*128]
            分支3:
                Conv:[1*1*12] stride=1 padding=0 active=ReLU out=[12*12*12]
                Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[14*14*32]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[12*12*256]
                Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(64+128+32+32)]=[12*12*256]
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_4c, self).__init__(name='Inception_V1_4c', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                64,
                [64, 128],
                [12, 32],
                32
            ]
    pass

#    Inception_4d
class Inception_V1_4d(Inception_V1):
    '''Inception_4d
        分为4支：
            分支1:
                Conv:[1*1*56] stride=1 padding=0 active=ReLU out=[12*12*56]
            分支2:
                Conv:[1*1*72] stride=1 padding=0 active=ReLU out=[12*12*72]
                Conv:[3*3*144] stride=1 padding=1 active=ReLU out=[14*14*144]
            分支3:
                Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[12*12*16]
                Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[12*12*32]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[12*12*256]
                Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(56+144+32+32)]=[12*12*264]  
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_4d, self).__init__(name='Inception_V1_4d', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                56,
                [72, 144],
                [16, 32],
                32
            ]
    pass

#    Inception_4e
class Inception_V1_4e(Inception_V1):
    '''Inception_4e
        分为4支：
            分支1:
                Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[12*12*128]
            分支2:
                Conv:[1*1*80] stride=1 padding=0 active=ReLU out=[12*12*80]
                Conv:[3*3*160] stride=1 padding=1 active=ReLU out=[12*12*160]
            分支3:
                Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[12*12*16]
                Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[12*12*64]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[12*12*264]
                Conv:[1*1*64] stride=1 padding=1 out=[12*12*64]
        out=[12*12*(128+160+64+64)]=[12*12*416] 
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_4e, self).__init__(name='Inception_V1_4e', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                128,
                [80, 160],
                [16, 64],
                64
            ]
    pass

#    Inception_5a
class Inception_V1_5a(Inception_V1):
    '''Inception_5a
        分为4支：
            分支1:
                Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[6*6*128]
            分支2:
                Conv:[1*1*80] stride=1 padding=0 active=ReLU out=[6*6*80]
                Conv:[3*3*160] stride=1 padding=1 active=ReLU out=[6*6*160]
            分支3:
                Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[6*6*16]
                Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[6*6*64]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[6*6*416]
                Conv:[1*1*64] stride=1 padding=1 out=[6*6*64]
        out=[6*6*(128+160+64+64)]=[6*6*416]   
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_5a, self).__init__(name='Inception_V1_5a', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                128,
                [80, 160],
                [16, 64],
                64
            ]
    pass

#    Inception_5b
class Inception_V1_5b(Inception_V1):
    '''Inception_5b
        分为4支：
            分支1:
                Conv:[1*1*192] stride=1 padding=0 active=ReLU out=[6*6*192]
            分支2:
                Conv:[1*1*96] stride=1 padding=0 active=ReLU out=[6*6*96]
                Conv:[3*3*192] stride=1 padding=1 active=ReLU out=[6*6*192]
            分支3:
                Conv:[1*1*24] stride=1 padding=0 active=ReLU out=[6*6*24]
                Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[6*6*64]
            分支4:
                max pooling:[3*3] stride=1 padding=1 out=[6*6*416]
                Conv:[1*1*64] stride=1 padding=1 out=[6*6*64]
        out=[7*7*(192+192+64+64)]=[6*6*512]
    '''
    def __init__(self, input_shape=None, output_shape=None, kernel_initializer=None, **kwargs):
        super(Inception_V1_5b, self).__init__(name='Inception_V1_5b', input_shape=input_shape, output_shape=output_shape, kernel_initializer=kernel_initializer, **kwargs)
        pass
    def conv_filters(self):
        return [
                192,
                [96, 192],
                [24, 64],
                [64]
            ]
    pass
