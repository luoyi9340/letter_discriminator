'''
AlexNet网络结构
    相比于LeNet，AlexNet主要改进：
        1 使用RELU作为激活单元。
        2 使用Dropout选择性忽略单个神经元，避免过拟合。
        3 选择最大池化，避免平均池化的平均化效果。

AlexNet网络（原生）
    输入：227 * 227 * 3（227*227 RGB图像）
    -----------------layer 1-------------------
    Conv:
        kernel_size=11*11*96 stride=4 padding=0
        out=55*55*96
    active:
        ReLU
    max pool:
        kernel_size=3*3 stride=2 padding=0
        out=27*27*96
    lrn:
        out=27*27*96
    -----------------layer 2-------------------    
    Conv:
        kernel_size=5*5*256 stride=1 padding=2
        out=27*27*256
    active:
        ReLU
    max pool:
        kernel_size=3*3 stride=2
        out:13*13*256
    lrn:
        out=13*13*256
    -----------------layer 3-------------------    
    Conv:
        kernel_size=3*3*384 stride=1 padding=1
        out=13*13*384
    active:
        ReLU
    Conv:
        kernel_size=3*3*384 stride=1 padding=1
        out=13*13*384
    active:
        ReLU
    -----------------layer 4-------------------  
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        out=13*13*256
    active:
        ReLU
    max pool:
        kernel_size=3*3 stride=2 padding=0
        out=6*6*256
    flatten:
        out=[9216]
    -----------------layer 5-------------------
    FC:
        w=[9216 * 4096]
        out=4096
    active:
        ReLU
    FC:
        w=[4096 * 4096]
        out=4096
    active:
        ReLU
    -----------------layer 6-------------------
    FC:
        w=[4096 * 1000]
        out=[1000]
    active:
        ReLU
    Softmax:
        out=[1000]个类别的概率
        
        
原生的AlexNet针对227*227*3的RGB图，而且最终分类有1000个。
而我们的测试图只有100*100*1的灰度，并且最终分类只有52个，若继续沿用原生AlexNet目测过拟合是一定的。
这里主要用AlexNet的改进点来改进LeNet网络：
    1 使用RELU作为激活单元。
    2 使用Dropout选择性忽略单个神经元，避免过拟合。
    3 选择最大池化，避免平均池化的平均化效果。
所以实际使用的AlexNet：
    输入：100 * 100 * 1（100*100灰度，归一化到0 ~ 1之间）
    -----------------layer 1-------------------
    Conv:
        kernel_size=4*4*32 stride=2 padding=0
        out=49*49*32
    active:
        ReLU
    max pooling:
        kernel_size=3*3 stride=2 padding=0
        out=23*23*32
    lrn:
        out=23*23*32（实际并没有做LRN，还是在输入之前做全局归一化吧）
    -----------------layer 2-------------------
    Conv:
        kernel_size=3*3*96 stride=1 padding=1
        out=23*23*96
    active:
        ReLU
    max pooling:
        kernel_size=3*3 stride=2 padding=0
        out=11*11*96
    lrn:
        out=11*11*96（实际并没有做LRN，还是在输入之前做全局归一化吧）
    -----------------layer 3-------------------
    Conv:
        kernel_size=3*3*192 stride=1 padding=1
        out=11*11*192
    active:
        ReLU
    Conv:
        kernel_size=3*3*192 stride=1 padding=1
        out=11*11*192
    active:
        ReLU
    -----------------layer 4-------------------
    Conv:
        kernel_size=3*3*128 stride=1 padding=1
        out=11*11*128
    active:
        ReLU
    max pooling:
        kernel_size=3*3 stride=2 padding=0
        out=5*5*128
    flatten:
        out=[1280]
    -----------------layer 5-------------------
    FC:
        w=[1280 * 512]
        out=[512]
    active:
        ReLU
    dropout:
        0.5
    FC:
        w[512 * 256]
        out=[256]
    active:
        ReLU
    dropout:
        0.5
    -----------------layer 6-------------------
    FC:
        w=[256 * 52]
        out=[52]
    active:
        Softmax
    


Created on 2020年12月17日

@author: irenebritney
'''
import tensorflow as tf

from model.abstract_model import AModel


#    AlexNet实现
class AlexNet(AModel):
    def __init__(self, learning_rate=0.9):
        super(AlexNet, self).__init__(learning_rate=learning_rate, name="AlexNet")
        pass
    
    #    子类必须指明梯度更新方式
    def optimizer(self, net, learning_rate=0.9):
        '''随机梯度下降'''
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #    子类必须指明损失函数
    def loss(self):
        '''交叉熵损失'''
        return tf.keras.losses.categorical_crossentropy
    #    子类必须指明评价方式
    def metrics(self):
        '''准确率评价'''
        return [tf.keras.metrics.Accuracy()]
    
    #    装配模型
    def assembling(self, net):
        #    layer 1
        net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=2, padding='valid', input_shape=(100, 100, 1), kernel_initializer='uniform'))
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=(49, 49, 32)))
        #    layer 2
        net.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(23, 23, 32), kernel_initializer='uniform'))
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=(23, 23, 96)))
        #    layer 3
        net.add(tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(11, 11, 96), kernel_initializer='uniform'))
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(11, 11, 192), kernel_initializer='uniform'))
        net.add(tf.keras.layers.Activation('relu'))
        #    layer 4
        net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(11, 11, 192), kernel_initializer='uniform'))
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid', input_shape=(11, 11, 128)))
        net.add(tf.keras.layers.Flatten())
        #    layer 5
        net.add(tf.keras.layers.Dense(512))
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.Dropout(0.5))
        net.add(tf.keras.layers.Dense(256))
        net.add(tf.keras.layers.Activation('relu'))
        net.add(tf.keras.layers.Dropout(0.5))
        #    layer 6
        net.add(tf.keras.layers.Dense(52))
        net.add(tf.keras.layers.Activation('softmax'))
        pass
    
    pass




