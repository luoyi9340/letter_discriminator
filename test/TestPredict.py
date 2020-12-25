# -*- coding: utf-8 -*-  
'''
Created on 2020年12月16日

@author: irenebritney
'''
import tensorflow as tf
import numpy as np

#    测试准确率函数
#    原以为tf.keras.metrics.Accuracy可以直接调用。想想确实可以，但就是不知道怎么调。。。
a = [1,2,3,4,5]
b = [1,2,3,4,6]
r = np.equal(a, b)
r = np.mean(r)
print(r)