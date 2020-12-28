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
from utils.Conf import LETTER


count = 10
batch_size=1
count_batch_train = int(count / batch_size * 0.8)
db_train = ds.load_tensor_db(x_filedir=LETTER.get_in_train(), 
                             y_filepath=LETTER.get_label_train(), 
                             count=count, 
                             batch_size=batch_size)


idx = random.randint(0, count)
i = 0
for x, y in db_train:
    y = y.numpy()[0]
    y = np.argmax(y)
    print(index_category(y))
    
    x = x.numpy()[0]
    x[x > 0] = 1
    plot.imshow(x, 'gray')
    plot.show()
    break
    pass

