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
from model.resnet.models import ResNet_18, ResNet_34, ResNet_50
from data import dataset
from utils.Conf import TRAIN, MODEL, LETTER


#    准备数据集
db_train = dataset.load_tensor_db(x_filedir=LETTER.get_in_train(), 
                                  y_filepath=LETTER.get_label_train(), 
                                  batch_size=TRAIN.get_train_batch_size(),
                                  count=LETTER.get_count_train())
db_val = dataset.load_tensor_db(x_filedir=LETTER.get_in_val(), 
                                y_filepath=LETTER.get_label_val(), 
                                batch_size=TRAIN.get_train_batch_size(),
                                count=LETTER.get_count_val())


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
# model = ResNet_18(learning_rate=TRAIN.get_learning_rate())
# net_weights_save_path = MODEL.get_resnet_18_save_weights_path()
model = ResNet_34(learning_rate=TRAIN.get_learning_rate())
net_weights_save_path = MODEL.get_resnet_34_save_weights_path()
# model = ResNet_50(learning_rate=TRAIN.get_learning_rate())
# net_weights_save_path = MODEL.get_resnet_50_save_weights_path()

model.show_info()

#    喂数据
his = model.train_tensor_db(db_train=db_train, 
                              db_val=db_val, 
                              batch_size=TRAIN.get_train_batch_size(),
                              epochs=TRAIN.get_epochs(),
                              auto_save_weights_after_traind=True,
                              auto_save_file_path=net_weights_save_path,
                              auto_tensorboard=True,
                              auto_tensorboard_dir=TRAIN.get_tensorboard_dir())
print(his)

#    保存模型参数（训练过程中会自动保存）
# model.save_model_weights(filepath=MODEL.get_resnet_18_save_weights_path())

