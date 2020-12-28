# -*- coding: utf-8 -*-  
'''
手写字母数据集数据集
    一共26（小写） + 26（大写） = 52个字母

Created on 2020年12月15日

@author: irenebritney
'''
import tensorflow as tf
import json
import matplotlib.pyplot as plot
import numpy as np

#    配置文件
from utils.Conf import LETTER, TRAIN
#    日志信息
from utils import LoggerFactory
from utils.Alphabet import alphabet, category_index


logger = LoggerFactory.get_logger('dataset')




#    单个数据one_hot
def one_hot(num):
    y = np.zeros(shape=len(alphabet), dtype=np.int8)
    y[num] = 1
    return y
#    db_generator
def db_generator(x_filedir, y_filepath, count, x_preprocess, y_preprocess):
    '''数据生成器
    '''
    c = 0
    for line in open(y_filepath, mode='r', encoding='utf-8'):
        #    控制读取数量
        if (c >= count): break
        c += 1
        #    读json格式
        d = json.loads(line)
        #    读取图片，并在末尾追加维度（整形为(100,100,1)），并做预处理
        x = plot.imread(x_filedir + '/' + d['filename'] + '.png', format)
        x = np.expand_dims(x, axis=-1)
        x = x_preprocess(x)
        #    y转数字，并做预处理
        y = category_index(d['letter'])
        y = y_preprocess(y)
        yield x, y
    pass
#    加载为tensorflow数据集
def load_tensor_db(x_filedir=None,
                   y_filepath=None, 
                   batch_size=32, 
                   count=200000,
                   x_preprocess=lambda x:(x - 0.5) * 2,
                   y_preprocess=one_hot):
    '''加载为tensor数据集
        @param x_filedir: 训练图片路径
        @param y_filepath: 标签文件路径
        @param batch_size: 批量大小
        @param count: 数据上限
        @param x_preprocess: 图片数据预处理
        @param y_preprocess: 标签数据预处理
        @return: tf.data.Dataset
    '''
    db = tf.data.Dataset.from_generator(lambda :db_generator(x_filedir, y_filepath, count, x_preprocess, y_preprocess), 
                                        output_types=(tf.float32, tf.int8),
                                        output_shapes=(tf.TensorShape([100, 100, 1]), tf.TensorShape([46]))).batch(batch_size)
    return db


