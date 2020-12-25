'''
定义模型的公共行为

Created on 2020年12月16日

@author: irenebritney
'''
import abc
import tensorflow as tf
import numpy as np

import random
from utils.Alphabet import index_category
import matplotlib.pyplot as plot

#    模型类的公共行为
class AModel(metaclass=abc.ABCMeta):
    '''模型的公共行为
        @method
        - 测试
        - 保存/载入模型参数
        - 训练
        @abstractmethod
        - 梯度下降方式
        - 损失函数
        - 评价标准
        - 装备模型
    '''
    def __init__(self, learning_rate=0.9, name="Model"):
        #    定义网络对象
        self._net = tf.keras.models.Sequential(name=name)
        
        #    装配网络模型
        self.assembling(self._net)
        
        #    编译网络
        self._net.compile(optimizer=self.optimizer(learning_rate=learning_rate, net=self._net), 
                          loss=self.loss(), 
                          metrics=self.metrics())
        
        pass
    
    #    测试
    def test(self, X_test, 
                    batch_size=32, 
                    verbose=1):
        '''跑测试集
            @param X_test: 测试集
            @param batch_size: 批量大小，默认32
            @param verbose: 默认日志级别（0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录）
            @return: 每个测试数据的最终分类索引
        '''
        pred = self._net.predict(X_test, batch_size=batch_size, verbose=verbose)
        pred = np.argmax(pred, axis=1)
        return pred
    
    #    测试准确率
    def test_accuracy(self, X_test, Y_test):
        '''跑测试集
            @param X_test: 测试集
            @param Y_test: 测试集标签
            @return: 准确率（0 ~ 1之间）
        '''
        pred = self.test(X_test, verbose=0)
        eq_res = np.equal(Y_test, pred)
        accuracy = np.mean(eq_res)
        return accuracy
    
    
    #    保存模型参数
    def save_model_weights(self, filepath):
        '''保存模型参数
            @param filepath: 保存文件路径（建议文件以.ckpt为后缀）
        '''
        self._net.save_weights(filepath, overwrite=True, save_format="h5")
        pass
    
    #    加载模型参数
    def load_model_weight(self, filepath):
        '''加载模型参数
            @param filepath: 加载模型路径
        '''
        self._net.load_weights(filepath)
        pass
    
    
    #    训练模型
    def train(self, X_train, Y_train, 
                    X_val, Y_val,
                    batch_size=32, 
                    epochs=5,
                    auto_save_weights_after_traind=True,
                    auto_save_file_path=None,
                    auto_learning_rate_schedule=True,
                    auto_tensorboard=True,
                    auto_tensorboard_dir=None
                    ):
        '''训练模型
            @param X_train: 训练集
            @param Y_train: 训练集标签
            @param X_val: 验证集（若验证集X，Y有一个为空或X，Y数量不对等则放弃验证）
            @param Y_val: 验证集标签（若验证集X，Y有一个为空或X，Y数量不对等则放弃验证）
            @param batch_size: 批量喂数据大小
            @param epochs: epoch次数
            @param auto_save_weights_after_traind: 是否在训练完成后自动保存（默认True）
            @param auto_save_file_path: 当auto_save_epoch为true时生效，保存参数文件path
            @param auto_learning_rate_schedule: 是否动态调整学习率
            @param auto_tensorboard: 是否开启tensorboard监听（一款tensorflow自带的可视化训练过程工具）
            @param auto_tensorboard_dir: tensorboard日志写入目录
            @return: history
        '''
        #    训练期间的回调
        callbacks = []
        #    如果需要每个epoch保存模型参数
        if (auto_save_weights_after_traind):
            auto_save_weights_callback = tf.keras.callbacks.ModelCheckpoint(filepath=auto_save_file_path,
                                                                            monitor="val_loss",         #    需要监视的值
                                                                            verbose=1,                      #    信息展示模式，0或1
                                                                            save_best_only=True,            #    当设置为True时，将只保存在验证集上性能最好的模型，一般我们都会设置为True. 
                                                                            model='auto',                   #    ‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
                                                                                                            #    例如:
                                                                                                            #        当监测值为val_acc时，模式应为max，
                                                                                                            #        当检测值为val_loss时，模式应为min。
                                                                                                            #        在auto模式下，评价准则由被监测值的名字自动推断。 
                                                                            save_weights_only=True,         #    若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等
                                                                            period=1                        #    CheckPoint之间的间隔的epoch数
                                                                            )
            callbacks.append(auto_save_weights_callback)
            pass
        #    如果需要在训练期间动态调整学习率
        if (auto_learning_rate_schedule):
            reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                        factor=0.1,             #    每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                                                        patience=1,             #    当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                                                        mode='auto',            #    ‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
                                                                        epsilon=0.0001,         #    阈值，用来确定是否进入检测值的“平原区” 
                                                                        cooldown=0,             #    学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                                                        min_lr=0.1              #    学习率的下限
                                                                        )
            callbacks.append(reduce_lr_on_plateau)
            pass
        #    如果需要在训练过程中开启tensorboard监听
        if (auto_tensorboard):
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=auto_tensorboard_dir,          #    tensorboard主目录
                                                         histogram_freq=1,                      #    对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 
                                                                                                #        如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
                                                         write_graph=True,                      #    是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True
                                                         write_grads=True,                      #    是否在 TensorBoard 中可视化梯度值直方图。 
                                                                                                #        histogram_freq 必须要大于 0
                                                         batch_size=batch_size,                 #    用以直方图计算的传入神经元网络输入批的大小
                                                         write_images=True,                     #    是否在 TensorBoard 中将模型权重以图片可视化，如果设置为True，日志文件会变得非常大
                                                         embeddings_freq=None,                  #    被选中的嵌入层会被保存的频率（在训练轮中）
                                                         embeddings_layer_names=None,           #    一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
                                                         embeddings_metadata=None,              #    一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字
                                                         embeddings_data=None,                  #    要嵌入在 embeddings_layer_names 指定的层的数据。 Numpy 数组（如果模型有单个输入）或 Numpy 数组列表（如果模型有多个输入）
                                                         update_freq='batch'                    #    'batch' 或 'epoch' 或 整数。
                                                                                                #        当使用 'batch' 时，在每个 batch 之后将损失和评估值写入到 TensorBoard 中。
                                                                                                #        同样的情况应用到 'epoch' 中。
                                                                                                #        如果使用整数，例如 10000，这个回调会在每 10000 个样本之后将损失和评估值写入到 TensorBoard 中。注意，频繁地写入到 TensorBoard 会减缓你的训练。
                                                         )
            callbacks.append(tensorboard)
            pass
        
        his = self._net.fit(x=X_train, y=Y_train,
                                batch_size=batch_size, 
                                epochs=epochs, 
                                verbose=1, 
                                validation_data=(X_val, Y_val),
                                callbacks=callbacks,
                                shuffle=False)
        return his


    '''以下是抽象方法定义
        python毕竟不是Java，找不到更多约束了。。。
    '''
    #    子类必须指明梯度更新方式
    @abc.abstractclassmethod
    def optimizer(self, net, learning_rate=0.9):
        pass
    #    子类必须指明损失函数
    @abc.abstractclassmethod
    def loss(self):
        pass
    #    子类必须指明评价方式
    @abc.abstractclassmethod
    def metrics(self):
        pass
    #    装配模型
    @abc.abstractclassmethod
    def assembling(self, net):
        pass
    pass


#    随机测试几个数据
def test_data(X, Y):
    print(len(X), len(Y))
    
    idx = random.randint(0, len(Y))
    x = X[idx]
    y = Y[idx]
    print(index_category(np.argmax(y)))
    plot.imshow(x, 'gray')
    plot.show()
    
    pass


