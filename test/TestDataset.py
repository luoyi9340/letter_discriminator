# -*- coding: utf-8 -*-  
'''
数据集测试

Created on 2020年12月15日

@author: irenebritney
'''
import random
import numpy as np
import json
import matplotlib.pyplot as plot
import tensorflow as tf

from data import dataset as ds
from utils.Alphabet import category_index, index_category


count = 10
batch_size=2
count_batch_train = int(count / batch_size * 0.8)
db = ds.load_tensor_db(count=10, batch_size=2)
#    前count_batch_train个batch用做训练
db_train = db.take(count_batch_train)
#    前count_batch_train个batch往后所有数据用作训练
db_val = db.skip(count_batch_train)


print("db_train:")
for x, y in db_train:
    print(x.shape, y.shape)

print("db_val:")
for x, y in db_val:
    print(x.shape, y.shape)