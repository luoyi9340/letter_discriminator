# -*- coding: utf-8 -*-  
'''
Created on 2020年12月15日

@author: irenebritney
'''


import yaml
import os


#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('LetterRecognizer')[0]
ROOT_PATH = ROOT_PATH + "LetterRecognizer"


#    取配置文件目录
CONF_PATH = ROOT_PATH + "/resources/conf.yml"
#    加载conf.yml配置文件
def load_conf_yaml():
    print('加载配置文件:' + CONF_PATH)
    f = open(CONF_PATH, 'r', encoding='utf-8')
    fr = f.read()
    
    c = yaml.load(fr)
    
    #    读取letter相关配置项
    letter = Letter(c['letter']['in'], c['letter']['count'], c['letter']['annotation'])
    #    读取train相关配置项
    train = Train(c['train']['rate_train'], c['train']['rate_val'], c['train']['rate_test'], c['train']['batch_size'], c['train']['tensorboard_dir'])
    model = Model(c['model']['lenet_save_weights_path'], 
                  c['model']['alexnet_save_weights_path'], 
                  c['model']['vggnet_save_weights_path'], 
                  c['model']['googlelenet_v1_save_weights_path'],
                  c['model']['googlelenet_v2_save_weights_path'],
                  c['model']['resnet_18_save_weights_path'],
                  c['model']['resnet_50_save_weights_path'])
    #    读取日志相关信息
    logs = Logs(c['logs'])
    return letter, train, model, logs

#    手写字母数据集。为了与Java的风格保持一致
class Letter:
    def __init__(self, letter_in="", letter_count=50000, letter_anno=""):
        self.__letter_in = letter_in
        self.__letter_count = letter_count
        self.__letter_anno = letter_anno
        pass
    
    def get_letter_in(self): return self.__letter_in
    def get_letter_count(self): return self.__letter_count
    def get_letter_anno(self): return self.__letter_anno
    pass
#    训练相关配置。为了与Java风格保持一致
class Train:
    def __init__(self, rate_train=0.8, rate_val=0.1, rate_test=0.1, batch_size=128, tensorboard_dir=""):
        self.__rate_train = rate_train
        self.__rate_val = rate_val
        self.__rate_test = rate_test
        self.__batch_size = batch_size
        self.__tensorboard_dir = tensorboard_dir
        pass
    
    def get_train_rate_train(self): return self.__rate_train
    def get_train_rate_val(self): return self.__rate_val
    def get_train_rate_test(self): return self.__rate_test
    def get_train_batch_size(self): return self.__batch_size
    def get_tensorboard_dir(self): return self.__tensorboard_dir
    pass
#    模型相关
class Model:
    def __init__(self, lenet_save_weights_path="", alexnet_save_weights_path="", vggnet_save_weights_path="", googlelenet_v1_save_weights_path="", googlelenet_v2_save_weights_path="", resnet_18_save_weights_path="", resnet_50_save_weights_path=""):
        self.__lenet_save_weights_path = lenet_save_weights_path
        self.__alexnet_save_weights_path = alexnet_save_weights_path
        self.__vggnet_save_weights_path = vggnet_save_weights_path
        self.__googlelenet_v1_save_weights_path = googlelenet_v1_save_weights_path
        self.__googlelenet_v2_save_weights_path = googlelenet_v2_save_weights_path
        self.__resnet_18_save_weights_path = resnet_18_save_weights_path
        self.__resnet_50_save_weights_path = resnet_50_save_weights_path
        pass
    def get_lenet_save_weights_path(self): return self.__lenet_save_weights_path
    def get_alexnet_save_weights_path(self): return self.__alexnet_save_weights_path
    def get_vggnet_save_weights_path(self): return self.__vggnet_save_weights_path
    def get_googlelenet_v1_save_weights_path(self): return self.__googlelenet_v1_save_weights_path
    def get_googlelenet_v2_save_weights_path(self): return self.__googlelenet_v2_save_weights_path
    def get_resnet_18_save_weights_path(self): return self.__resnet_18_save_weights_path
    def get_resnet_50_save_weights_path(self): return self.__resnet_50_save_weights_path
    pass
#    log相关配置（只有该配置信息是原始dict）
class Logs:
    def __init__(self, conf_dict):
        self.__dict = conf_dict
        pass
    def get_logs_dict(self): return self.__dict
    pass


LETTER, TRAIN, MODEL, LOGS = load_conf_yaml()