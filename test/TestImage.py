# -*- coding: utf-8 -*-  
'''
Created on 2020年12月15日

@author: luoyi
'''
import matplotlib.pyplot as plot
import tensorflow as tf


path = "/Users/irenebritney/Desktop/vcode/letter/0a0a4596-4e75-444f-ba51-896cf423021c.png"
arr = plot.imread(path, "png")
print(arr)

x = tf.io.read_file(path)
x = tf.image.decode_png(x, channels=1)
x = tf.cast(x, dtype=tf.float32) / 255.
print(x.shape)