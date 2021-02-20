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
    letter = Letter(c['letter']['in_train'], c['letter']['count_train'], c['letter']['label_train'],
                    c['letter']['in_val'], c['letter']['count_val'], c['letter']['label_val'],
                    c['letter']['in_test'], c['letter']['count_test'], c['letter']['label_test'])
    #    读取train相关配置项
    train = Train(c['train']['rate_train'], 
                  c['train']['rate_val'], 
                  c['train']['rate_test'], 
                  c['train']['batch_size'], 
                  c['train']['tensorboard_dir'],
                  c['train']['epochs'],
                  c['train']['learning_rate'])
    model = Model(c['model']['lenet_save_weights_path'], 
                  c['model']['alexnet_save_weights_path'], 
                  c['model']['vggnet_save_weights_path'], 
                  c['model']['googlelenet_v1_save_weights_path'],
                  c['model']['googlelenet_v2_save_weights_path'],
                  c['model']['resnet_18_save_weights_path'],
                  c['model']['resnet_34_save_weights_path'],
                  c['model']['resnet_50_save_weights_path'],
                  c['model']['densenet_121'])
    #    读取日志相关信息
    logs = Logs(c['logs'])
    return letter, train, model, logs

#    手写字母数据集。为了与Java的风格保持一致
class Letter:
    def __init__(self, in_train="", count_train=50000, label_train="", in_val="", count_val=10000, label_val="", in_test="", count_test=10000, label_test=""):
        self.__in_train = in_train
        self.__count_train = count_train
        self.__label_train = label_train
        
        self.__in_val = in_val
        self.__count_val = count_val
        self.__label_val = label_val
        
        self.__in_test = in_test
        self.__count_test = count_test
        self.__label_test = label_test
        pass
    def get_in_train(self): return self.__in_train
    def get_count_train(self): return self.__count_train
    def get_label_train(self): return self.__label_train
    
    def get_in_val(self): return self.__in_val
    def get_count_val(self): return self.__count_val
    def get_label_val(self): return self.__label_val    
    
    def get_in_test(self): return self.__in_test
    def get_count_test(self): return self.__count_test
    def get_label_test(self): return self.__label_test
    pass
#    训练相关配置。为了与Java风格保持一致
class Train:
    def __init__(self, rate_train=0.8, rate_val=0.1, rate_test=0.1, batch_size=128, tensorboard_dir="", epochs=5, learning_rate=0.1):
        self.__rate_train = rate_train
        self.__rate_val = rate_val
        self.__rate_test = rate_test
        self.__batch_size = batch_size
        self.__tensorboard_dir = tensorboard_dir
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        pass
    
    def get_train_rate_train(self): return self.__rate_train
    def get_train_rate_val(self): return self.__rate_val
    def get_train_rate_test(self): return self.__rate_test
    def get_train_batch_size(self): return self.__batch_size
    def get_tensorboard_dir(self): return self.__tensorboard_dir
    def get_epochs(self): return self.__epochs
    def get_learning_rate(self): return self.__learning_rate
    pass
#    模型相关
class Model:
    def __init__(self, lenet_save_weights_path="", alexnet_save_weights_path="", vggnet_save_weights_path="", googlelenet_v1_save_weights_path="", googlelenet_v2_save_weights_path="", resnet_18_save_weights_path="", resnet_34_save_weights_path="", resnet_50_save_weights_path="",
                 densenet_121=''):
        self.__lenet_save_weights_path = lenet_save_weights_path
        self.__alexnet_save_weights_path = alexnet_save_weights_path
        self.__vggnet_save_weights_path = vggnet_save_weights_path
        self.__googlelenet_v1_save_weights_path = googlelenet_v1_save_weights_path
        self.__googlelenet_v2_save_weights_path = googlelenet_v2_save_weights_path
        self.__resnet_18_save_weights_path = resnet_18_save_weights_path
        self.__resnet_34_save_weights_path = resnet_34_save_weights_path
        self.__resnet_50_save_weights_path = resnet_50_save_weights_path
        self.__densenet_121 = densenet_121
        pass
    def get_lenet_save_weights_path(self): return self.__lenet_save_weights_path
    def get_alexnet_save_weights_path(self): return self.__alexnet_save_weights_path
    def get_vggnet_save_weights_path(self): return self.__vggnet_save_weights_path
    def get_googlelenet_v1_save_weights_path(self): return self.__googlelenet_v1_save_weights_path
    def get_googlelenet_v2_save_weights_path(self): return self.__googlelenet_v2_save_weights_path
    def get_resnet_18_save_weights_path(self): return self.__resnet_18_save_weights_path
    def get_resnet_34_save_weights_path(self): return self.__resnet_34_save_weights_path
    def get_resnet_50_save_weights_path(self): return self.__resnet_50_save_weights_path
    def get_densenet_121(self): return self.__densenet_121
    pass
#    log相关配置（只有该配置信息是原始dict）
class Logs:
    def __init__(self, conf_dict):
        self.__dict = conf_dict
        pass
    def get_logs_dict(self): return self.__dict
    pass


LETTER, TRAIN, MODEL, LOGS = load_conf_yaml()