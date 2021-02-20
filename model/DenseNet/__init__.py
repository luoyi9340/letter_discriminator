# -*- coding: utf-8 -*-  
'''
    DenseNet模型
        与ResNet想法类似，不过这里的每层layer输出会在通道维度上叠加之前各层的输出。达到特征重用的目的
    
    一些超参数：
        growth rate：超参数，DenseBlock层的输出通道数。一般设置为12，Dense Block每一层都输出这么多个通道
                        一下简写为k
        compression rate：超参数，Transition层的通道压缩系数，取值(0,1]
                        以下简写为c
    
    DenseBlock层：
        不改变特征图尺寸，仅在通道维度叠加之前各层输出
        网络结构：
        Conv: [1*1*4k] strides=1 padding=0 norm=BN active=ReLU out=[H * W * 4k]
        Conv: [3*3*k] strides=1 padding=1 norm=BN active=ReLU out=[H * W * k]
    
    Transition层：
        连接相邻两个DenseBlock层，起到降维和收缩特征图尺寸的作用
        网络结构：
        Conv: k=[1*1*（k*c）] strides=1 padding=0 norm=BN active=ReLU out=[H * W * (k*c)]
        avg_pooling: k=[2,2] strides=2 padding=0 out=[H/2 * W/2 * (k*c)]
        
    网络结构：
    growth_rate = 12
    compression_rate = 0.5
    
    
    
    DenseNet-121
    ---------- 输入 ----------
    输入：[224 * 224 * 3]
    ---------- layer1 ----------
    Conv: [3 * 3 * growth_rate*2] strides=2 padding=1 norm=BN active=ReLU out=[112 * 112 * growth_rate*2]
    max_pooling: [3 * 3] strides=2 padding=1 out=[56 * 56 * growth_rate*2]
    ---------- layer2 ----------
    DenseBlock1:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[56 * 56 * growth_rate]
        concat: 
        times: 6
        out: [56 * 56 * kdb1=growth_rate*8]
    Transition1:
        Conv: k=[1*1*（kdb1*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * (kdb1*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[28 * 28 * (kdb1*compression_rate)]
    out=[28 * 28 * k1=(kdb1*compression_rate)]
    ---------- layer2 ----------
    DenseBlock2:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[28 * 28 * growth_rate]
        concat: 
        times: 12
        out: [28 * 28 * kdb2=k1+growth_rate*12]
    Transition2:
        Conv: k=[1*1*（kdb2*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * (kdb2*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[14 * 14 * (kdb2*compression_rate)]
    out=[14 * 14 * k2=(kdb2*compression_rate)]
    ---------- layer3 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[14 * 14 * growth_rate]
        concat: 
        times: 24
        out: [14 * 14 * kdb3=k2+growth_rate*24]
    Transition2:
        Conv: k=[1*1*（kdb3*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * (kdb3*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[7 * 7 * (kdb3*compression_rate)]
    out=[7 * 7 * k3=(kdb3*compression_rate)]
    ---------- layer4 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[7 * 7 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[7 * 7 * growth_rate]
        concat: 
        times: 16
        out: [7 * 7 * kdb4=k3+growth_rate*16]
    out=[7 * 7 * k4=k3+growth_rate*16]
    ---------- layer5 ----------
    avg_pooling: [7 * 7] strides=1 padding=0 out=[1 * 1 * k4]
    fc: [1000] active=Softmax out=[1000]（假设有1000个分类）
    
    
    DenseNet-169
    ---------- 输入 ----------
    输入：[224 * 224 * 3]
    ---------- layer1 ----------
    Conv: [3 * 3 * growth_rate*2] strides=2 padding=1 norm=BN active=ReLU out=[112 * 112 * growth_rate*2]
    max_pooling: [3 * 3] strides=2 padding=1 out=[56 * 56 * growth_rate*2]
    ---------- layer2 ----------
    DenseBlock1:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[56 * 56 * growth_rate]
        concat: 
        times: 6
        out: [56 * 56 * kdb1=growth_rate*8]
    Transition1:
        Conv: k=[1*1*（kdb1*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * (kdb1*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[28 * 28 * (kdb1*compression_rate)]
    out=[28 * 28 * k1=(kdb1*compression_rate)]
    ---------- layer2 ----------
    DenseBlock2:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[28 * 28 * growth_rate]
        concat: 
        times: 12
        out: [28 * 28 * kdb2=k1+growth_rate*12]
    Transition2:
        Conv: k=[1*1*（kdb2*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * (kdb2*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[14 * 14 * (kdb2*compression_rate)]
    out=[14 * 14 * k2=(kdb2*compression_rate)]
    ---------- layer3 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[14 * 14 * growth_rate]
        concat: 
        times: 32
        out: [14 * 14 * kdb3=k2+growth_rate*32]
    Transition2:
        Conv: k=[1*1*（kdb3*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * (kdb3*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[7 * 7 * (kdb3*compression_rate)]
    out=[7 * 7 * k3=(kdb3*compression_rate)]
    ---------- layer4 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[7 * 7 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[7 * 7 * growth_rate]
        concat: 
        times: 32
        out: [7 * 7 * kdb4=k3+growth_rate*32]
    out=[7 * 7 * k4=kdb4]
    ---------- layer5 ----------
    avg_pooling: [7 * 7] strides=1 padding=0 out=[1 * 1 * k4]
    fc: [1000] active=Softmax out=[1000]（假设有1000个分类）
    
    
        DenseNet-121
    ---------- 输入 ----------
    输入：[224 * 224 * 3]
    ---------- layer1 ----------
    Conv: [3 * 3 * growth_rate*2] strides=2 padding=1 norm=BN active=ReLU out=[112 * 112 * growth_rate*2]
    max_pooling: [3 * 3] strides=2 padding=1 out=[56 * 56 * growth_rate*2]
    ---------- layer2 ----------
    DenseBlock1:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[56 * 56 * growth_rate]
        concat: 
        times: 6
        out: [56 * 56 * kdb1=growth_rate*8]
    Transition1:
        Conv: k=[1*1*（kdb1*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * (kdb1*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[28 * 28 * (kdb1*compression_rate)]
    out=[28 * 28 * k1=(kdb1*compression_rate)]
    ---------- layer2 ----------
    DenseBlock2:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[28 * 28 * growth_rate]
        concat: 
        times: 12
        out: [28 * 28 * kdb2=k1+growth_rate*12]
    Transition2:
        Conv: k=[1*1*（kdb2*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * (kdb2*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[14 * 14 * (kdb2*compression_rate)]
    out=[14 * 14 * k2=(kdb2*compression_rate)]
    ---------- layer3 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[14 * 14 * growth_rate]
        concat: 
        times: 48
        out: [14 * 14 * kdb3=k2+growth_rate*48]
    Transition2:
        Conv: k=[1*1*（kdb3*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * (kdb3*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[7 * 7 * (kdb3*compression_rate)]
    out=[7 * 7 * k3=(kdb3*compression_rate)]
    ---------- layer4 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[7 * 7 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[7 * 7 * growth_rate]
        concat: 
        times: 32
        out: [7 * 7 * kdb4=k3+growth_rate*32]
    out=[7 * 7 * k4=k3+growth_rate*32]
    ---------- layer5 ----------
    avg_pooling: [7 * 7] strides=1 padding=0 out=[1 * 1 * k4]
    fc: [1000] active=Softmax out=[1000]（假设有1000个分类）
    
    
        DenseNet-121
    ---------- 输入 ----------
    输入：[224 * 224 * 3]
    ---------- layer1 ----------
    Conv: [3 * 3 * growth_rate*2] strides=2 padding=1 norm=BN active=ReLU out=[112 * 112 * growth_rate*2]
    max_pooling: [3 * 3] strides=2 padding=1 out=[56 * 56 * growth_rate*2]
    ---------- layer2 ----------
    DenseBlock1:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[56 * 56 * growth_rate]
        concat: 
        times: 6
        out: [56 * 56 * kdb1=growth_rate*8]
    Transition1:
        Conv: k=[1*1*（kdb1*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[56 * 56 * (kdb1*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[28 * 28 * (kdb1*compression_rate)]
    out=[28 * 28 * k1=(kdb1*compression_rate)]
    ---------- layer2 ----------
    DenseBlock2:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[28 * 28 * growth_rate]
        concat: 
        times: 12
        out: [28 * 28 * kdb2=k1+growth_rate*12]
    Transition2:
        Conv: k=[1*1*（kdb2*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[28 * 28 * (kdb2*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[14 * 14 * (kdb2*compression_rate)]
    out=[14 * 14 * k2=(kdb2*compression_rate)]
    ---------- layer3 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[14 * 14 * growth_rate]
        concat: 
        times: 64
        out: [14 * 14 * kdb3=k2+growth_rate*64]
    Transition2:
        Conv: k=[1*1*（kdb3*compression_rate）] strides=1 padding=0 norm=BN active=ReLU out=[14 * 14 * (kdb3*compression_rate)]
        avg_pooling: [2 * 2] strides=2 padding=0 out=[7 * 7 * (kdb3*compression_rate)]
    out=[7 * 7 * k3=(kdb3*compression_rate)]
    ---------- layer4 ----------
    DenseBlock3:
        Conv: [1 * 1 * growth_rate*4] strides=1 padding=0 norm=BN active=ReLU out=[7 * 7 * growth_rate*4]
        Conv: [3 * 3 * growth_rate] strides=1 padding=1 norm=BN active=ReLU out=[7 * 7 * growth_rate]
        concat: 
        times: 48
        out: [7 * 7 * kdb4=k3+growth_rate*48]
    out=[7 * 7 * k4=k3+growth_rate*48]
    ---------- layer5 ----------
    avg_pooling: [7 * 7] strides=1 padding=0 out=[1 * 1 * k4]
    fc: [1000] active=Softmax out=[1000]（假设有1000个分类）
'''