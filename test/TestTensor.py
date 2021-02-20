# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年2月20日
'''
import tensorflow as tf


a = tf.range(16)
a = tf.reshape(a, shape=[1, 16])
res = tf.split(a, num_or_size_splits=8, axis=-1)
l = [i for i in range(8)]
for a_, l_ in zip(res, l):
    print(a_)
    print(l_)
    print()
    pass


