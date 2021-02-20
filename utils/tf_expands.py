# -*- coding: utf-8 -*-  
'''
tf扩展

@author: luoyi
Created on 2021年2月21日
'''
import tensorflow as tf
from math import floor, ceil


#    BN + ReLU + Conv
class CombinationConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 name='CombinationConv2D',
                 norm=tf.keras.layers.BatchNormalization(),
                 active=tf.keras.layers.ReLU(),
                 kernel_size=[3,3],
                 filters=16,
                 strides=1,
                 padding='VALID',
                 kernel_initializer=tf.initializers.he_normal(),
                 bias_initializer=tf.keras.initializers.zeros(),
                 **kwargs):
        super(CombinationConv2D, self).__init__(name=name, **kwargs)
        
        self._net = tf.keras.models.Sequential(name=name)
        #    BN
        if (norm is None): norm = tf.keras.layers.BatchNormalization()
        self._net.add(norm)
        
        #    active
        if (active is None): active = tf.keras.layers.ReLU()
        self._net.add(active)
        
        #    conv
        if (isinstance(padding, int)):
            self._net.add(tf.keras.layers.ZeroPadding2D(padding=padding))
            self._net.add(tf.keras.layers.Conv2D(name=name + '_conv',
                                                 kernel_size=kernel_size,
                                                 filters=filters,
                                                 strides=strides,
                                                 padding='VALID',
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer
                                                 ))
            pass
        else:
            self._net.add(tf.keras.layers.Conv2D(name=name + '_conv',
                                                 kernel_size=kernel_size,
                                                 filters=filters,
                                                 strides=strides,
                                                 padding=padding,
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer
                                                 ))
            pass
        pass
    
    def call(self, inputs, **kwargs):
        return self._net(inputs)
    pass


a = tf.reshape(tf.range(10), shape=(1, 10))
part_ratio = 0.59
p1 = round(a.shape[-1] * part_ratio)
p2 = a.shape[-1] - p1
[p1, p2] = tf.split(a, num_or_size_splits=[p1,p2], axis=-1)
print(p1)
print(p2)
