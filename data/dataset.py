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


#    加载全部原始数据
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




#    加载为tensorflow数据集
def load_tensor_db(X, Y, batch_size=TRAIN.get_train_batch_size()):
    '''加载为tensor数据集
    
    @param X: 训练数据集
    @param Y: 训练数据标签集
    @param batch_size: 批量大小
    @return: tf.data.Dataset
    '''
    db_train = tf.data.Dataset.from_tensor_slices((X, Y))
    #    shuffle打乱数据顺序，map对数据预处理，batch设置批量大小
    db_train.shuffle(1000).map(preprocess).batch(batch_size)
    return db_train
#    tensorflow数据集每条数据前置处理
def preprocess(x, y):
    '''数据前置处理
        1 从图片路径加载为像素矩阵
        2 像素矩阵
    '''
    #    从图片路径加载为像素矩阵
    x = tf.io.read_file(x)
    x = tf.image.decode_png(x, channels=1)              #    本次数据全是灰度模式
    x = tf.cast(x, dtype=tf.float32) / 255.             #    数据缩放到0~1之间
    return x, y



