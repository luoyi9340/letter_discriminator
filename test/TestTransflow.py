'''
Tensorflow测试类

Created on 2020年12月21日

@author: irenebritney
'''
import tensorflow as tf
import numpy as np

print(tf.version.VERSION)


a = tf.convert_to_tensor(np.ones((3, 1), dtype=np.int8))
b = tf.convert_to_tensor(np.ones((3, 1), dtype=np.int8))
c = tf.convert_to_tensor(np.ones((3, 1), dtype=np.int8))
d = tf.concat([a, b, c], axis=1)
# print(d.shape == (3, 3))
# 
# if (d.shape != (3, 3)):
#     raise Exception("d.shape:" + str(d.shape) + " not equal:" + str((3, 3)))
# 
# a = [
#         1,
#         [1, 2],
#         2
#     ]
# print(a)

x = tf.convert_to_tensor(np.ones(shape=(3, 3, 3), dtype=np.int8))
x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last').call(x)
print(x.numpy().shape)

# a = tf.convert_to_tensor(np.ones(shape=(2, 3, 3), dtype=np.int8))
# b = tf.convert_to_tensor(np.ones(shape=(1, 3, 3), dtype=np.int8))
# c = tf.keras.layers.add([a, b])
# print(c.numpy())

a = tf.convert_to_tensor(np.ones(shape=(1, 3, 3, 1), dtype=np.int8))
a = tf.keras.layers.ZeroPadding2D(padding=1).call(a)
print(a.numpy().shape)



