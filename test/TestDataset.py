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
batch_size=1
count_batch_train = int(count / batch_size * 0.8)
db = ds.load_tensor_db(count=count, batch_size=batch_size)
#    前count_batch_train个batch用做训练
db_train = db.take(count_batch_train)
#    前count_batch_train个batch往后所有数据用作训练
db_val = db.skip(count_batch_train)


idx = random.randint(0, count)
i = 0
for x, y in db:
    i += 1
    if (i <= idx): continue

        
    y = y.numpy()[0]
    y = np.argmax(y)
    print(index_category(y))
    
    x = x.numpy()[0]
    plot.imshow(x, 'gray')
    plot.show()
    break
    pass

