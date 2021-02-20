# -*- coding: utf-8 -*-  
'''
ResNeXt网络
    借鉴GoogleLeNet的思路，增加网络宽度

ResNeXtBlock_C
    参数：
        group = 分组卷积个数
        base_filters = 每个分组卷积核个数
        out_filters = 输出通道数
        conv_strides = [1, 1]每个分组/每个卷积层步长
                        若conv_strides[1] == 1，则输出特征图尺寸不变
                        若conv_strides[1] == 2，则输出特征图尺寸缩小为原1/2
    输入：X = [H * W * C1]
    Conv: [1*1*(group * base_filters)] strides=1 padding=0 norm=BN active=ReLU out=[H * W * (group * base_filters)]
    ---------- 分组卷积 ----------
    分支1:
        输入：[H, W, 0:base_filters]
        Conv: [1*1*base_filters] strides=1 padding=0 norm=BN active=ReLU out=[H * W * base_filters]
        Conv:  [3*3*base_filters] strides=2 padding=1 norm=BN active=ReLU out=[H/2 * W/2 * base_filters]
    分支2:
        输入：[H, W, base_filters:2*base_filters]
        Conv: [1*1*base_filters] strides=1 padding=0 norm=BN active=ReLU out=[H * W * base_filters]
        Conv:  [3*3*base_filters] strides=2 padding=1 norm=BN active=ReLU out=[H/2 * W/2 * base_filters]
    ... 循环N次 ...
    Concat:
        在通道维度追加上述N个分支的结果
        out = [H/2 * W/2 * (group * base_filters)]
    维度调整：
        Y = Conv: [1*1*out_filters] strides=1 padding=0 norm=BN active=ReLU out=[H/2 * W/2 * out_filters]
    残差块
        Y = Y + f(X)
        f(x)为Conv[1*1*out_filters]线性变换，让X的尺寸与Y的尺寸/通道数一致，方便求和操作
    ---------- 输出 ----------
    输出：[H/2 * W/2 * out_filters]
    
    
ResNeXt-50网络结构：
    输入：[224 * 224 * 3]
    ---------- layer1 ----------
    Conv: [7*7*64] s=2 p=3 norm=BN active=ReLU out=[112 * 112 * 64]
    max_pooling: [3 * 3] s=2 p=0 out=[56 * 56 * 64]
    ---------- layer2 ----------
    ResNeXtBlock:
        base_filters = 4
        group = 32
        strides = [1, 1]
        out_filters = 256
        out = [56 * 56 * 256]
    times: 3
    out = [56 * 56 * 256]
    ---------- layer3 ----------
    ResNeXtBlock:
        base_filters = 8
        group = 32
        strides = [1, 2]
        out_filters = 512
        out = [28 * 28 * 512]
    ResNeXtBlock:
        base_filters = 8
        group = 32
        strides = [1, 1]
        out_filters = 512
        out = [28 * 28 * 512]
    times: 3
    out = [28 * 28 * 512]
    ---------- layer4 ----------
    ResNeXtBlock:
        base_filters = 16
        group = 32
        strides = [1, 2]
        out_filters = 1024
        out = [14 * 14 * 1024]
    ResNeXtBlock:
        base_filters = 16
        group = 32
        strides = [1, 1]
        out_filters = 1024
        out = [14 * 14 * 1024]
    times: 5
    out = [14 * 14 * 1024]
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
    fc: [1000] active=Softmax out=[1000] （假定分类数=1000）
    
'''
