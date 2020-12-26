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


#    加载全部原始数据（慎用，次方法很吃内存）
def load_all_anno(count=LETTER.get_letter_count()):
    '''加载全部原始数据
        从配置文件的letter.anno加载所有原始标记数据
        @return: X（图片本机绝对路径）, Y（图片标签编码0~25对应a~z，26~51对应A~Z）
    '''
    i = 0
    X = []
    Y = []
    for line in open(LETTER.get_letter_anno(), 'r', encoding='utf-8'):
        if (i >= count): break
        i = i + 1
        
        d = json.loads(line)
        X.append(LETTER.get_letter_in() + "/" + d['filename'] + ".png")
        Y.append(category_index(d['letter']))
        pass
    
    logger.info("load original data:" + str(i))
    return np.array(X), np.array(Y)
#    原始数据x加载为图片像素矩阵
def load_image(X, preprocess=lambda x: x / 255.):
    '''原始数据x的绝对路径加载为图片像素矩阵，并且归一化到0~1之间
        @param X: 图片绝对路径list
        @param preprocess: 像素矩阵后置处理（默认归到0均值，0~1之间）
        @return: 每个绝对路径对应的归一化后的图片像素矩阵
    '''
    new_X = []
    for i in range(len(X)):
        mat = plot.imread(X[i])
        mat = preprocess(mat)
        #    整形为(w, h, 1)，灰度模式追加一个通道
        mat = np.expand_dims(mat, -1)
#         print(mat.shape)
        new_X.append(mat)
        pass
    return np.array(new_X)
#    原始数据y加载为one-hot编码
def load_one_hot(Y):
    '''索引位概率为1，其他为0
    '''
    new_Y = []
    for i in range(len(Y)):
        #    字母表长度即为分类个数
        y = np.zeros(shape=(len(alphabet)), dtype=np.float32)
        y[Y[i]] = 1
        new_Y.append(y)
        pass
    return np.array(new_Y)


#    按比例切分数据集
def original_db_distribution(X, Y,
                            rate_train=TRAIN.get_train_rate_train(), 
                             rate_val=TRAIN.get_train_rate_val(), 
                             rate_test=TRAIN.get_train_rate_test()):
    '''按比例切分数据集
    
    @param rate_train: 训练集占比
    @param rate_val: 验证集占比
    @param rate_test: 测试集占比
    @return: X_train, Y_train
                X_val, Y_val
                X_test, Y_test
    '''
    count = len(Y)      #    以标注数据量为准
    
    train_start = 0
    train_end = int(rate_train * count)
    X_train, Y_train = None, None
    if (rate_train > 0):
        X_train = X[train_start : train_end]
        Y_train = Y[train_start : train_end]
        pass
    
    val_start = train_end
    val_end = int(val_start + rate_val * count)
    X_val, Y_val = None, None
    if (rate_val > 0):
        X_val = X[val_start : val_end]
        Y_val = Y[val_start : val_end]
        pass
    
    test_start = val_end
    test_end = int(val_end + rate_test * count)
    X_test, Y_test = None, None
    if (rate_test > 0):
        X_test = X[test_start : test_end]
        Y_test = Y[test_start : test_end]
        pass
    
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test



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
def load_tensor_db(x_filedir=LETTER.get_letter_in(),
                   y_filepath=LETTER.get_letter_anno(), 
                   batch_size=TRAIN.get_train_batch_size(), 
                   count=LETTER.get_letter_count(),
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


