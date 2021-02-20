# -*- coding: utf-8 -*-  
'''
ResNeXt模型

@author: luoyi
Created on 2021年2月20日
'''
import tensorflow as tf


from model.abstract_model import AModel
from model.ResNeXt.part import ResNeXtBlock_C


#    ResNeXt50模型
class ResNeXt_50(AModel):
    '''
        输入：[100 * 100 * 1]
        ---------- layer1 ----------
        Conv: [2*2*32] s=2 p=0 norm=BN active=ReLU out=[50 * 50 * 32]
        ---------- layer2 ----------
        ResNeXtBlock:
            base_filters = 4
            group = 32
            strides = [1, 1]
            out_filters = 256
            out = [50 * 50 * 256]
        times: 3
        out = [50 * 50 * 256]
        ---------- layer3 ----------
        ResNeXtBlock:
            base_filters = 8
            group = 32
            strides = [1, 2]
            out_filters = 512
            out = [25 * 25 * 512]
        ResNeXtBlock:
            base_filters = 8
            group = 32
            strides = [1, 1]
            out_filters = 512
            out = [25 * 25 * 512]
        times: 3
        out = [25 * 25 * 512]
        ---------- layer4 ----------
        ResNeXtBlock:
            base_filters = 16
            group = 32
            strides = [1, 2]
            out_filters = 1024
            out = [13 * 13 * 1024]
        ResNeXtBlock:
            base_filters = 16
            group = 32
            strides = [1, 1]
            out_filters = 1024
            out = [13 * 13 * 1024]
        times: 5
        out = [13 * 13 * 1024]
        ---------- layer5 ----------
        ResNeXtBlock:
            base_filters = 32
            group = 32
            strides = [1, 2]
            out_filters = 2048
            out = [7 * 7 * 2048]
        ResNeXtBlock:
            base_filters = 32
            group = 32
            strides = [1, 1]
            out_filters = 2048
            out = [7 * 7 * 2048]
        times: 2
        out = [7 * 7 * 2048]
        ---------- layer6 ----------
        avg_pooling: [7*7] strides=1 padding=0 out=[1 * 1 * 2048]
        fc: [46] active=Softmax out=[46] 
    '''
    def __init__(self, 
                 learning_rate=0.01,
                 group=32,
                 base_filters=0.5,
                 batch_size=32,
                 input_shape=(100, 100, 1)):
        
        self._group = group
        self._base_filters = base_filters
        self.batch_size = batch_size
        self._input_shape = input_shape
        
        super(ResNeXt_50, self).__init__(name='ResNeXt_50', learning_rate=learning_rate)
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
        
        
        #    layer1
        l1_in = self._input_shape
        net.add(tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=[2,2],
                                       strides=2,
                                       padding='VALID',
                                       input_shape=l1_in,
                                       kernel_initializer=tf.initializers.he_normal(),
                                       bias_initializer=tf.initializers.zeros()))
        
        #    layer2
        l2_in = (50, 50, 32)
        l2_out = (50, 50, 256)
        l2_base_filters = 4
        net.add(ResNeXtBlock_C(name='ResNeXt_layer11', base_filters=l2_base_filters, is_down_sample=False,
                               input_shape=l2_in, output_shape=l2_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer12', base_filters=l2_base_filters, is_down_sample=False,
                               input_shape=l2_out, output_shape=l2_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer13', base_filters=l2_base_filters, is_down_sample=False,
                               input_shape=l2_out, output_shape=l2_out))
        
        #    layer3
        l3_in = (50, 50, 256)
        l3_out = (25, 25, 512)
        l3_base_filters = 8
        net.add(ResNeXtBlock_C(name='ResNeXt_layer21', base_filters=l3_base_filters, is_down_sample=True,
                               input_shape=l3_in, output_shape=l3_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer22', base_filters=l3_base_filters, is_down_sample=False,
                               input_shape=l3_out, output_shape=l3_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer23', base_filters=l3_base_filters, is_down_sample=False,
                               input_shape=l3_out, output_shape=l3_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer24', base_filters=l3_base_filters, is_down_sample=False,
                               input_shape=l3_out, output_shape=l3_out))
        
        #    layer4
        l4_in = (25, 25, 512)
        l4_out = (13, 13, 1024)
        l4_base_filters = 16
        net.add(ResNeXtBlock_C(name='ResNeXt_layer31', base_filters=l4_base_filters, is_down_sample=True,
                               input_shape=l4_in, output_shape=l4_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer32', base_filters=l4_base_filters, is_down_sample=False,
                               input_shape=l4_out, output_shape=l4_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer33', base_filters=l4_base_filters, is_down_sample=False,
                               input_shape=l4_out, output_shape=l4_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer34', base_filters=l4_base_filters, is_down_sample=False,
                               input_shape=l4_out, output_shape=l4_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer35', base_filters=l4_base_filters, is_down_sample=False,
                               input_shape=l4_out, output_shape=l4_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer36', base_filters=l4_base_filters, is_down_sample=False,
                               input_shape=l4_out, output_shape=l4_out))
        
        #    layer5
        l5_in = (13, 13, 1024)
        l5_out = (7, 7, 2048)
        l5_base_filters = 32
        net.add(ResNeXtBlock_C(name='ResNeXt_layer41', base_filters=l5_base_filters, is_down_sample=True,
                               input_shape=l5_in, output_shape=l5_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer42', base_filters=l5_base_filters, is_down_sample=False,
                               input_shape=l5_out, output_shape=l5_out))
        net.add(ResNeXtBlock_C(name='ResNeXt_layer43', base_filters=l5_base_filters, is_down_sample=False,
                               input_shape=l5_out, output_shape=l5_out))
        
        #    layer6
        net.add(tf.keras.layers.AvgPool2D(pool_size=[7,7], strides=1, padding='VALID'))
        net.add(tf.keras.layers.Flatten())
        net.add(tf.keras.layers.Dense(units=46))
        net.add(tf.keras.layers.Softmax())
        pass
    pass


