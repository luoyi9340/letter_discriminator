# -*- coding: utf-8 -*-  
'''
训练模型

Created on 2020年12月16日

@author: irenebritney
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('LetterRecognizer')[0]
ROOT_PATH = ROOT_PATH + "LetterRecognizer"
sys.path.append(ROOT_PATH)


from model.LeNet import LeNet_5
from model.AlexNet import AlexNet
from model.googlelenet.v1.GoogleLeNet import GoogleLeNet_V1
from model.googlelenet.v2.GoogleLeNet import GoogleLeNet_V2
from model.VGGNet import VGGNet_16
from model.resnet.models import ResNet_18, ResNet_50
from data import dataset
from utils.Conf import TRAIN, MODEL


#    准备数据集
X, Y = dataset.load_all_anno()
X = dataset.load_image(X, preprocess=lambda x:x)             #    将x归到均值0方差1的分布中
Y = dataset.load_one_hot(Y)
X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset.original_db_distribution(X, Y, rate_train=0.95, rate_val=0.05, rate_test=0)
print("X_train.len:" + str(len(X_train)), 
      " Y_train.len:" + str(len(Y_train)), 
      " X_val.len:" + str(len(X_val)),
      " Y_val.len:" + str(len(Y_val)))
# db_train = dataset.load_tensor_db(X_train, Y_train)
# db_val = dataset.load_tensor_db(X_val, Y_val)
# db_test = dataset.load_tensor_db(X_test, Y_test)


#    初始化模型
#    初始化LeNet-5（LeNet-5并没有训练完成，太慢了。。。）
# model = LeNet_5(learning_rate=0.9)
# net_weights_save_path=MODEL.get_lenet_save_weights_path()

#    初始化AlexNet
# model = AlexNet(learning_rate=0.9)
# net_weights_save_path = MODEL.get_alexnet_save_weights_path()

#    初始化VGG16
# model = VGGNet_16(learning_rate=0.2)
# net_weights_save_path = MODEL.get_vggnet_save_weights_path()

#    初始化GoogleLeNet_V1
# model = GoogleLeNet_V1()
# net_weights_save_path = MODEL.get_googlelenet_v1_save_weights_path()
# model = GoogleLeNet_V2()
# net_weights_save_path = MODEL.get_googlelenet_v2_save_weights_path()

#    初始化ResNet模型
# model = ResNet_18(learning_rate=0.9)
# net_weights_save_path = MODEL.get_resnet_18_save_weights_path()
model = ResNet_18()
net_weights_save_path = MODEL.get_resnet_50_save_weights_path()


#    喂数据
his = model.train(X_train=X_train, Y_train=Y_train, 
                  X_val=X_val, Y_val=Y_val, 
                  batch_size=TRAIN.get_train_batch_size(),
                  auto_save_weights_after_traind=True,
                  auto_save_file_path=net_weights_save_path,
                  auto_tensorboard=True,
                  auto_tensorboard_dir=TRAIN.get_tensorboard_dir())
print(his)

#    保存模型参数（训练过程中会自动保存）
# model.save_model_weights(filepath=MODEL.get_resnet_18_save_weights_path())

