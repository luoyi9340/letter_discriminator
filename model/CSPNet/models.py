# -*- coding: utf-8 -*-  
'''
CSPDenseNet网络

@author: luoyi
Created on 2021年2月21日
'''
import tensorflow as tf

from model.abstract_model import AModel
from model.CSPNet.part_dense_net import CSPDenseBlockLayer, CSPTransitionLayer
from math import ceil, floor


#    DenseNet-121
class CSPDenseNet_121(AModel):
    '''
        ---------- 输入 ----------
        输入：[100 * 100 * 1]
        ---------- layer1 ----------
        Conv: [2 * 2 * growth_rate*2] strides=2 padding=0 norm=BN active=ReLU out=[50 * 50 * growth_rate*2]
        ---------- layer2 ----------
        DenseBlock1:
            Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[50 * 50 * growth_rate*4]
            Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[50 * 50 * growth_rate]
            concat: 
            times: 6
            out: [50 * 50 * kdb1=growth_rate*8]
        Transition1:
            Conv: k=[1*1*（kdb1*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[50 * 50 * (kdb1*compression_rate)]
            avg_pooling: [2 * 2] strides=2 padding=0 out=[25 * 25 * (kdb1*compression_rate)]
        out=[25 * 25 * k1=(kdb1*compression_rate)]
        ---------- layer3 ----------
        DenseBlock2:
            Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[25 * 25 * growth_rate*4]
            Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[25 * 25 * growth_rate]
            concat: 
            times: 12
            out: [25 * 25 * kdb2=k1+growth_rate*12]
        Transition2:
            Conv: k=[1*1*（kdb2*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[25 * 25 * (kdb2*compression_rate)]
            avg_pooling: [2 * 2] strides=2 padding=0 out=[13 * 13 * (kdb2*compression_rate)]
        out=[14 * 14 * k2=(kdb2*compression_rate)]
        ---------- layer4 ----------
        DenseBlock3:
            Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[13 * 13 * growth_rate*4]
            Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[13 * 13 * growth_rate]
            concat: 
            times: 24
            out: [13 * 13 * kdb3=k2+growth_rate*24]
        Transition2:
            Conv: k=[1*1*（kdb3*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[13 * 13 * (kdb3*compression_rate)]
            avg_pooling: [2 * 2] strides=2 padding=0 out=[7 * 7 * (kdb3*compression_rate)]
        out=[7 * 7 * k3=(kdb3*compression_rate)]
        ---------- layer5 ----------
        DenseBlock3:
            Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[7 * 7 * growth_rate*4]
            Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[7 * 7 * growth_rate]
            concat: 
            times: 16
            out: [7 * 7 * kdb4=k3+growth_rate*16]
        out=[7 * 7 * k4=k3+growth_rate*16]
        ---------- layer6 ----------
        avg_pooling: [7 * 7] strides=1 padding=0 out=[1 * 1 * k4]
        fc: [46] active=Softmax out=[46]
    '''
    def __init__(self, 
                 learning_rate=0.01,
                 growth_rate=12,
                 compression_rate=0.5,
                 part_ratio=0.5,
                 batch_size=32,
                 input_shape=(100, 100, 1)):
        self._growth_rate = growth_rate
        self._compression_rate = compression_rate
        self._part_ratio = part_ratio
        self._learning_rate = learning_rate
        self._input_shape = input_shape
        self.batch_size = batch_size
        
        super(CSPDenseNet_121, self).__init__(name='CSPDenseNet_121', learning_rate=learning_rate)
        pass

    #    子类必须指明梯度更新方式
    def optimizer(self, net, learning_rate=0.9):
        return tf.optimizers.Adam(learning_rate=learning_rate)

    #    子类必须指明损失函数
    def loss(self):
        return tf.losses.categorical_crossentropy

    #    子类必须指明评价方式
    def metrics(self):
        return [tf.metrics.CategoricalAccuracy()]
    
    #    装配模型
    def assembling(self, net):
        #    输入
        input_shape = self._input_shape
        growth_rate = self._growth_rate
        compression_rate = self._compression_rate
        
        l1_in = input_shape
        l1_out = (floor(l1_in[0]/2), floor(l1_in[1]/2), growth_rate * 2)
        net.add(tf.keras.layers.Conv2D(name='conv1',
                                       input_shape=l1_in,
                                       filters=growth_rate * 2,
                                       kernel_size=[2, 2],
                                       strides=2,
                                       padding='VALID',
                                       kernel_initializer=tf.initializers.he_normal(),
                                       bias_initializer=tf.initializers.zeros()))
        
        #    layer2
        l2_d_in = (l1_out[0], l1_out[1], l1_out[2])
        l2_d_out = (l2_d_in[0], l2_d_in[1], l2_d_in[2] + growth_rate*6)
        net.add(CSPDenseBlockLayer(name='dense_block_2',
                                   growth_rate=self._growth_rate,
                                   part_ratio=self._part_ratio,
                                   num=6,
                                   input_shape=l2_d_in,
                                   output_shape=l2_d_out))
        l2_t_in = (l2_d_out[0], l2_d_out[1], l2_d_out[2])
        l2_t_out = (floor(l2_t_in[0]/2), floor(l2_t_in[1]/2), ceil(l2_t_in[2] * compression_rate))
        net.add(CSPTransitionLayer(name='transition_2',
                                   compression_rate=compression_rate,
                                   input_shape=l2_t_in,
                                   output_shape=l2_t_out
                                   ))
        
        #    layer3
        l3_d_in = (l2_t_out[0], l2_t_out[1], l2_t_out[2])
        l3_d_out = (l3_d_in[0], l3_d_in[1], l3_d_in[2] + growth_rate*12)
        net.add(CSPDenseBlockLayer(name='dense_block_3',
                                   growth_rate=self._growth_rate,
                                   part_ratio=self._part_ratio,
                                   num=12,
                                   input_shape=l3_d_in,
                                   output_shape=l3_d_out))
        l3_t_in = (l3_d_out[0], l3_d_out[1], l3_d_out[2])
        l3_t_out = (floor(l3_t_in[0]/2), floor(l3_t_in[1]/2), ceil(l3_t_in[2] * compression_rate))
        net.add(CSPTransitionLayer(name='transition_3',
                                   compression_rate=compression_rate,
                                   input_shape=l3_t_in,
                                   output_shape=l3_t_out
                                   ))
        
        #    layer4
        l4_d_in = (l3_t_out[0], l3_t_out[1], l3_t_out[2])
        l4_d_out = (l4_d_in[0], l4_d_in[1], l4_d_in[2] + growth_rate*24)
        net.add(CSPDenseBlockLayer(name='dense_block_4',
                                   growth_rate=self._growth_rate,
                                   part_ratio=self._part_ratio,
                                   num=24,
                                   input_shape=l4_d_in,
                                   output_shape=l4_d_out))
        l4_t_in = (l4_d_out[0], l4_d_out[1], l4_d_out[2])
        l4_t_out = (floor(l4_t_in[0]/2), floor(l4_t_in[1]/2), int(l4_t_in[2] * compression_rate))
        net.add(CSPTransitionLayer(name='transition_4',
                                   compression_rate=compression_rate,
                                   input_shape=l4_t_in,
                                   output_shape=l4_t_out
                                   ))
        
        #    layer5
        l5_d_in = (l4_t_out[0], l4_t_out[1], l4_t_out[2])
        l5_d_out = (l5_d_in[0], l5_d_in[1], l5_d_in[2] + growth_rate*16)
        net.add(CSPDenseBlockLayer(name='dense_block_5',
                                   growth_rate=self._growth_rate,
                                   part_ratio=self._part_ratio,
                                   num=16,
                                   input_shape=l5_d_in,
                                   output_shape=l5_d_out))
        
        #    layer6
        net.add(tf.keras.layers.AvgPool2D(pool_size=[l5_d_out[0], l5_d_out[1]],
                                          strides=1,
                                          padding='VALID'))
        net.add(tf.keras.layers.Flatten())
        net.add(tf.keras.layers.Dense(units=46))
        net.add(tf.keras.layers.Softmax())
        pass

    pass