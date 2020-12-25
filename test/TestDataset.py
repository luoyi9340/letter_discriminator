'''
数据集测试

Created on 2020年12月15日

@author: irenebritney
'''
import random
import numpy as np
import matplotlib.pyplot as plot

from data import dataset as ds
from utils.Alphabet import category_index, index_category


X, Y = ds.load_all_anno()
X_, Y_, X_, Y_, X, Y = ds.original_db_distribution(X, Y)
Y_label = ds.load_one_hot(Y)
# print(len(Y))

# X_train, Y_train, X_val, Y_val, X_test, Y_test = ds.original_db_distribution(X, Y)
# print(len(Y_train), len(Y_val), len(Y_test))

# db_train = ds.load_tensor_db(X_train, Y_train)
# print(db_train)

idx = random.randint(0, len(X))

print(index_category(Y[idx]))
print(Y_label[idx], index_category(np.argmax(Y_label[idx])))
x = ds.load_image([X[idx]], preprocess=lambda x:x)[0]
print(x.shape)
# x = np.squeeze(x, axis=0)
# x = np.squeeze(x, axis=3)
plot.imshow(x, 'gray')
plot.show()


