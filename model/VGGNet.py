'''
VGG网络
在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核
这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果

VGG16网络结构（原生）
    输入：224*224*3
    -----------------layer 1-------------------
    Conv:
        kernel_size=3*3*64 stride=1 padding=1
        out=224*224*64
    active
        ReLU
    Conv:
        kernel_size=3*3*64 stride=1 padding=1
        out=224*224*64
    active
        ReLU
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=112*112*64
    -----------------layer 2-------------------
    Conv:
        kernel_size=3*3*128 stride=1 padding=1
        out=112*112*128
    active
        ReLU
    Conv:
        kernel_size=3*3*128 stride=1 padding=1
        out=112*112*128
    active
        ReLU
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=56*56*128
    -----------------layer 3-------------------
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        out=56*56*256
    active
        ReLU
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        out=56*56*256
    active
        ReLU
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        out=56*56*256
    active
        ReLU
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=28*28*256
    -----------------layer 4-------------------
    Conv:
        kernel_size=3*3*512 stride=1 padding=1
        out=28*28*512
    active
        ReLU
    Conv:
        kernel_size=3*3*512 stride=1 padding=1
        out=28*28*512
    active
        ReLU
    Conv:
        kernel_size=3*3*512 stride=1 padding=1
        out=28*28*512
    active
        ReLU
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=14*14*512
    -----------------layer 5-------------------
    Conv:
        kernel_size=3*3*512 stride=1 padding=1
        out=14*14*512
    active
        ReLU
    Conv:
        kernel_size=3*3*512 stride=1 padding=1
        out=14*14*512
    active
        ReLU
    Conv:
        kernel_size=3*3*512 stride=1 padding=1
        out=14*14*512
    active
        ReLU
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=7*7*512
    -----------------layer 6-------------------
    FC:
        w=[25099 * 4096]
        out=[4096]
    active:
        ReLU
    dropout:
        0.5
    FC:
        w=[4096 * 4096]
        out=[4096]
    active:
        ReLU
    dropout:
        0.5
    -----------------layer 7-------------------
    FC:
        w=[4096 * 1000]
        out=[4096]
    active:
        Softmax
        
        
简化版VGG16：
    输入：100*100*1
    -----------------layer 1-------------------
    Conv:
        kernel_size=3*3*32 stride=1 padding=1
        active=ReLU
        out=100*100*32
    Conv:
        kernel_size=3*3*32 stride=1 padding=1
        active=ReLU
        out=100*100*32
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=50*50*32
    -----------------layer 2-------------------
    Conv:
        kernel_size=3*3*64 stride=1 padding=1
        active=ReLU
        out=50*50*64
    Conv:
        kernel_size=3*3*64 stride=1 padding=1
        active=ReLU
        out=50*50*64
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=25*25*64
    -----------------layer 3-------------------
    Conv:
        kernel_size=3*3*128 stride=1 padding=1
        active=ReLU
        out=25*25*128
    Conv:
        kernel_size=3*3*128 stride=1 padding=1
        active=ReLU
        out=25*25*128
    Conv:
        kernel_size=3*3*128 stride=1 padding=1
        active=ReLU
        out=25*25*128
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=12*12*128
    -----------------layer 4-------------------
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        active=ReLU
        out=12*12*256
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        active=ReLU
        out=12*12*256
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        active=ReLU
        out=12*12*256
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=6*6*256    
    -----------------layer 5-------------------   
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        active=ReLU
        out=6*6*256
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        active=ReLU
        out=6*6*256
    Conv:
        kernel_size=3*3*256 stride=1 padding=1
        active=ReLU
        out=6*6*256
    max pool:
        kernel_size=2*2 stride=2 padding=0
        out=3*3*256      
    -----------------layer 6------------------
    FC:
        w=[2304 * 256]
        active=ReLU
        out=[256]
    dropout:
        0.5
    FC:
        w=[256 * 256]
        active=ReLU
        out=[256]
    dropout:
        0.5
    -----------------layer 7------------------
    FC:
        w=[256 * 52]
        active=Softmax
        out=[52]
    
        


Created on 2020年12月19日

@author: luoyi
'''
import tensorflow as tf

from model.abstract_model import AModel



#    VGG16网络
class VGGNet_16(AModel):
    def __init__(self, learning_rate=0.9):
        super(VGGNet_16, self).__init__(learning_rate=learning_rate, name="VGGNet-16")
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
        net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', input_shape=(100, 100, 1), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', input_shape=(100, 100, 32), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(100, 100, 32)))
        #    layer 2
        net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(50, 50, 32), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(50, 50, 64), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(50, 50, 64)))
        #    layer 3
        net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(25, 25, 64), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(25, 25, 128), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(25, 25, 128), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(25, 25, 128)))
        #    layer 4
        net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(12, 12, 128), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(12, 12, 256), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(12, 12, 256), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(12, 12, 256)))
        #    layer 5
        net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(6, 6, 256), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(6, 6, 256), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', input_shape=(6, 6, 256), activation='relu', kernel_initializer='uniform'))
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(6, 6, 256)))
        net.add(tf.keras.layers.Flatten())
        #    layer 6
        net.add(tf.keras.layers.Dense(256, activation='relu'))
        net.add(tf.keras.layers.Dropout(0.5))
        net.add(tf.keras.layers.Dense(256, activation='relu'))
        net.add(tf.keras.layers.Dropout(0.5))
        #    layer 7
        net.add(tf.keras.layers.Dense(52, activation='softmax'))
        pass
    
    
    pass

