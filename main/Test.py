# -*- coding: utf-8 -*-  
'''
验证训练的模型

Created on 2020年12月17日

@author: irenebritney
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('LetterRecognizer')[0]
ROOT_PATH = ROOT_PATH + "LetterRecognizer"
sys.path.append(ROOT_PATH)

import matplotlib.pyplot as plot
import numpy as np

from utils.Conf import MODEL, LETTER, TRAIN
from model.LeNet import LeNet_5
from model.AlexNet import AlexNet
from model.VGGNet import VGGNet_16
from model.resnet.models import ResNet_18, ResNet_34, ResNet_50
from data import dataset
from utils.Alphabet import index_category


#    加载模型参数
# model = LeNet_5()
# model.load_model_weight(MODEL.get_lenet_save_weights_path())

# model = AlexNet()
# model.load_model_weight(MODEL.get_alexnet_save_weights_path())

# model = VGGNet_16()
# model.load_model_weight(MODEL.get_vggnet_save_weights_path())

# model = ResNet_18()
# model.load_model_weight(MODEL.get_resnet_18_save_weights_path())

# model = ResNet_34()
# model.load_model_weight(MODEL.get_resnet_34_save_weights_path())

model = ResNet_50()
model.load_model_weight(MODEL.get_resnet_50_save_weights_path())

#    加载测试数据
X_test, Y_test = dataset.load_all_anno()
X_test = dataset.load_image(X_test)

# Y_pred = model.test(X_test)
# print(Y_pred, index_category(Y_pred))
# print(Y_test, index_category(Y_test[0]))

print('test_accuracy:' + str(model.test_accuracy(X_test, Y_test)))

