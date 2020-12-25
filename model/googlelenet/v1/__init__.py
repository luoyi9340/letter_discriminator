# -*- coding: utf-8 -*-  
'''
GoogleLeNet网络
简介：
    GoogLeNet,作为 ILSVRC-2014的分类和检测任务的冠军，
    相比于当年分类任务第二名VGG Net的对于小卷积层（3x3）的简单堆叠，GoogLeNet提出更具创意的Inception模块，
    虽然网络结构比较复杂，但是模型参数量却降低了，仅为AlexNet的1/12,而VGGNet的参数量却是AlexNet的3倍，但模型精度却比VGG要跟高。


Inception_V1模块：
    即使用不同大小的卷积核同时卷积，最后再把结果拼接起来。达到不同的感受野尽量充分的提取特征。
    结构如下：
        1 输入：M*N*C
        2 并行卷积核
            - 分支1：
                Conv[1*1*C11 1stride=1 padding=0 activation=relu] out=M*N*C11
                out=M*N*C11
            - 分支2：
                Conv[1*1*C21 stride=1 padding=0 activation=relu] out=M*N*C21
                Conv[3*3*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                out=M*N*C22
            - 分支3：
                Conv[1*1*C31 stride=1 padding=0 activation=relu] out=M*N*C31
                Conv[5*5*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                out=M*N*C32
            - 分支4：
                max pooling[3*3 stride=1 padding=1] out=M*N*C
                Conv[1*1*C41 stride=1 padding=0 activation=relu] out=M*N*C41
                out=M*N*C41
            out=[ M * N * (C11 + C22 + C32 + C41)]
    这么做的好处：
        1 使用不同大小的卷积核充分提取特征，适应不同尺度的检测目标
        2 Conv[1*1] + Conv[5*5]代替原来直接Conv[5*5]，达到减少参数的目录。
            例如：
                输入：[M*N*128]
                Conv[5*5*256]
                    参数数量=128*5*5*256=819200
                Conv[1*1*32] + Conv[5*5*256]
                    参数数量=128*1*1*32 + 32*5*5*256=208896
                参数数量只有原来的1/4
    总之，Inception模型的好处是既能增加网络的深度和宽度，又不会增加计算量，而且稀疏连接的方式还能有助于减少过拟合。
    


GoogleLeNet_V1网络结构（原生）
    输入：224 * 224 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=7*7*64 stride=2 padding=3
        active=ReLU
        out=112*112*64
    max pooling:
        kernel_size=3*3 stride=2 padding=1
        out=56*56*64
    -----------------layer 2-------------------
    Conv:
        kernel_size=3*3*192 stride=1 padding=1
        active=ReLU
        out=56*56*192
    max pooling:
        kernel_size=3*3 stride=2 padding=1
        out=28*28*192
    -----------------Inception 3a-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[28*28*64]
        分支2:
            Conv:[1*1*96] stride=1 padding=1 active=ReLU out=[28*28*96]
            Conv:[3*3*128] stride=1 padding=1 active=ReLU out=[28*28*128]
        分支3:
            Conv:[1*1*16] stride=1 padding=1 active=ReLU out=[28*28*16]
            Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[28*28*32]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[28*28*192]
            Conv:[1*1*32] stride=1 padding=1 out=[28*28*32]
        out=[28*28*(64+128+32+32)]=[28*28*256]
    -----------------Inception 3b-------------------
    分为4支：
        分支1:
            Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[28*28*128]
        分支2:
            Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[28*28*128]
            Conv:[3*3*192] stride=1 padding=1 active=ReLU out=[28*28*192]
        分支3:
            Conv:[1*1*32] stride=1 padding=0 active=ReLU out=[28*28*32]
            Conv:[5*5*96] stride=1 padding=2 active=ReLU out=[28*28*96]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[28*28*256]
            Conv:[1*1*64] stride=1 padding=1 out=[28*28*64]
        out=[28*28*(128+192+96+64)]=[28*28*480]
    -----------------layer 3-------------------
    max pooling:
        kernel_size=3*3 stride=2 padding=1
        out=14*14*480
    -----------------Inception 4a-------------------
    分为4支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 active=ReLU out=[14*14*192]
        分支2:
            Conv:[1*1*96] stride=1 padding=0 active=ReLU out=[14*14*96]
            Conv:[3*3*208] stride=1 padding=1 active=ReLU out=[14*14*208]
        分支3:
            Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[14*14*16]
            Conv:[5*5*48] stride=1 padding=2 active=ReLU out=[14*14*48]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[14*14*480]
            Conv:[1*1*64] stride=1 padding=1 out=[14*14*64]
        out=[28*28*(192+208+48+64)]=[14*14*512]
    -----------------Inception 4b-------------------
    分为4支：
        分支1:
            Conv:[1*1*160] stride=1 padding=0 active=ReLU out=[14*14*160]
        分支2:
            Conv:[1*1*112] stride=1 padding=0 active=ReLU out=[14*14*112]
            Conv:[3*3*224] stride=1 padding=1 active=ReLU out=[14*14*224]
        分支3:
            Conv:[1*1*24] stride=1 padding=0 active=ReLU out=[14*14*24]
            Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[14*14*64]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[14*14*512]
            Conv:[1*1*64] stride=1 padding=1 out=[14*14*64]
        out=[28*28*(160+224+64+64)]=[14*14*512]    
    -----------------Inception 4c-------------------
    分为4支：
        分支1:
            Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[14*14*128]
        分支2:
            Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[14*14*128]
            Conv:[3*3*256] stride=1 padding=1 active=ReLU out=[14*14*256]
        分支3:
            Conv:[1*1*24] stride=1 padding=0 active=ReLU out=[14*14*24]
            Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[14*14*64]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[14*14*512]
            Conv:[1*1*64] stride=1 padding=1 out=[14*14*64]
        out=[28*28*(128+256+64+64)]=[14*14*512]      
    -----------------Inception 4d-------------------
    分为4支：
        分支1:
            Conv:[1*1*112] stride=1 padding=0 active=ReLU out=[14*14*112]
        分支2:
            Conv:[1*1*144] stride=1 padding=0 active=ReLU out=[14*14*144]
            Conv:[3*3*288] stride=1 padding=1 active=ReLU out=[14*14*288]
        分支3:
            Conv:[1*1*32] stride=1 padding=0 active=ReLU out=[14*14*32]
            Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[14*14*64]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[14*14*512]
            Conv:[1*1*64] stride=1 padding=1 out=[14*14*64]
        out=[28*28*(112+288+64+64)]=[14*14*528]        
    -----------------Inception 4e-------------------
    分为4支：
        分支1:
            Conv:[1*1*256] stride=1 padding=0 active=ReLU out=[14*14*256]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 active=ReLU out=[14*14*160]
            Conv:[3*3*320] stride=1 padding=1 active=ReLU out=[14*14*320]
        分支3:
            Conv:[1*1*32] stride=1 padding=0 active=ReLU out=[14*14*32]
            Conv:[5*5*128] stride=1 padding=2 active=ReLU out=[14*14*128]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[14*14*528]
            Conv:[1*1*128] stride=1 padding=1 out=[14*14*128]
        out=[14*14*(256+320+128+128)]=[14*14*832]     
    -----------------layer 4-------------------
    max pooling:
        kernel_size=3*3 stride=2 padding=1
        out=7*7*832
    -----------------Inception 5a-------------------
    分为4支：
        分支1:
            Conv:[1*1*256] stride=1 padding=0 active=ReLU out=[7*7*256]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 active=ReLU out=[7*7*160]
            Conv:[3*3*320] stride=1 padding=1 active=ReLU out=[7*7*320]
        分支3:
            Conv:[1*1*32] stride=1 padding=0 active=ReLU out=[7*7*32]
            Conv:[5*5*128] stride=1 padding=2 active=ReLU out=[7*7*128]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[7*7*832]
            Conv:[1*1*128] stride=1 padding=1 out=[7*7*128]
        out=[7*7*(256+320+128+128)]=[7*7*832]     
    -----------------Inception 5b-------------------
    分为4支：
        分支1:
            Conv:[1*1*384] stride=1 padding=0 active=ReLU out=[7*7*384]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 active=ReLU out=[7*7*192]
            Conv:[3*3*384] stride=1 padding=1 active=ReLU out=[7*7*384]
        分支3:
            Conv:[1*1*48] stride=1 padding=0 active=ReLU out=[7*7*48]
            Conv:[5*5*128] stride=1 padding=2 active=ReLU out=[7*7*128]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[7*7*832]
            Conv:[1*1*128] stride=1 padding=1 out=[7*7*128]
        out=[7*7*(384+384+128+128)]=[7*7*1024]     
    -----------------layer 5-------------------
    avg pooling:
        kernel_size=[7*7*1]
        out=[1*1*1024]
    faltten:
        out=[1024]
    FC:
        w=[1024 * 1000]
        out=[1000]
        active=Softmax
    
    
    

GoogleLeNet_V1（简化版）
    输入：100 * 100 * 1
    -----------------layer 1-------------------
    Conv:
        kernel_size=3*3*32 stride=1 padding=1
        active=ReLU
        out=100*100*32
    max pooling:
        kernel_size=2*2 stride=2 padding=0
        out=50*50*32
    -----------------layer 2-------------------
    Conv:
        kernel_size=3*3*96 stride=1 padding=1
        active=ReLU
        out=50*50*96
    max pooling:
        kernel_size=2*2 stride=2 padding=0
        out=25*25*96
    -----------------Inception 3a-------------------
    分为4支：
        分支1:
            Conv:[1*1*32] stride=1 padding=0 active=ReLU out=[25*25*32]
        分支2:
            Conv:[1*1*48] stride=1 padding=1 active=ReLU out=[25*25*48]
            Conv:[3*3*64] stride=1 padding=1 active=ReLU out=[25*25*64]
        分支3:
            Conv:[1*1*8] stride=1 padding=1 active=ReLU out=[25*25*8]
            Conv:[5*5*16] stride=1 padding=2 active=ReLU out=[25*25*16]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[25*25*96]
            Conv:[1*1*16] stride=1 padding=1 out=[25*25*16]
        out=[25*25*(32+64+16+16)]=[28*28*128]
    -----------------Inception 3b-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[25*25*64]
        分支2:
            Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[25*25*64]
            Conv:[3*3*96] stride=1 padding=1 active=ReLU out=[25*25*96]
        分支3:
            Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[25*25*16]
            Conv:[5*5*48] stride=1 padding=2 active=ReLU out=[25*25*48]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[25*25*128]
            Conv:[1*1*32] stride=1 padding=1 out=[25*25*32]
        out=[25*25*(64+96+48+32)]=[25*25*240]
    -----------------layer 3-------------------
    max pooling:
        kernel_size=3*3 stride=2 padding=0
        out=12*12*240
    -----------------Inception 4a-------------------
    分为4支：
        分支1:
            Conv:[1*1*96] stride=1 padding=0 active=ReLU out=[12*12*96]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 active=ReLU out=[12*12*48]
            Conv:[3*3*104] stride=1 padding=1 active=ReLU out=[14*14*104]
        分支3:
            Conv:[1*1*8] stride=1 padding=0 active=ReLU out=[12*12*8]
            Conv:[5*5*24] stride=1 padding=2 active=ReLU out=[12*12*24]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[12*12*240]
            Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(96+104+24+32)]=[12*12*256]
    -----------------Inception 4b-------------------
    分为4支：
        分支1:
            Conv:[1*1*80] stride=1 padding=0 active=ReLU out=[12*12*80]
        分支2:
            Conv:[1*1*56] stride=1 padding=0 active=ReLU out=[12*12*56]
            Conv:[3*3*112] stride=1 padding=1 active=ReLU out=[12*12*112]
        分支3:
            Conv:[1*1*12] stride=1 padding=0 active=ReLU out=[12*12*12]
            Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[12*12*32]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[12*12*256]
            Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(80+112+32+32)]=[12*12*256]
    -----------------Inception 4c-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[12*12*64]
        分支2:
            Conv:[1*1*64] stride=1 padding=0 active=ReLU out=[12*12*64]
            Conv:[3*3*128] stride=1 padding=1 active=ReLU out=[12*12*128]
        分支3:
            Conv:[1*1*12] stride=1 padding=0 active=ReLU out=[12*12*12]
            Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[14*14*32]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[12*12*256]
            Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(64+128+32+32)]=[12*12*256]
    -----------------Inception 4d-------------------
    分为4支：
        分支1:
            Conv:[1*1*56] stride=1 padding=0 active=ReLU out=[12*12*56]
        分支2:
            Conv:[1*1*72] stride=1 padding=0 active=ReLU out=[12*12*72]
            Conv:[3*3*144] stride=1 padding=1 active=ReLU out=[14*14*144]
        分支3:
            Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[12*12*16]
            Conv:[5*5*32] stride=1 padding=2 active=ReLU out=[12*12*32]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[12*12*256]
            Conv:[1*1*32] stride=1 padding=1 out=[12*12*32]
        out=[12*12*(56+144+32+32)]=[12*12*264]  
    -----------------Inception 4e-------------------
    分为4支：
        分支1:
            Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[12*12*128]
        分支2:
            Conv:[1*1*80] stride=1 padding=0 active=ReLU out=[12*12*80]
            Conv:[3*3*160] stride=1 padding=1 active=ReLU out=[12*12*160]
        分支3:
            Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[12*12*16]
            Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[12*12*64]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[12*12*264]
            Conv:[1*1*64] stride=1 padding=1 out=[12*12*64]
        out=[12*12*(128+160+64+64)]=[12*12*416] 
    -----------------layer 4-------------------
    max pooling:
        kernel_size=2*2 stride=2 padding=0
        out=6*6*416
    -----------------Inception 5a-------------------
    分为4支：
        分支1:
            Conv:[1*1*128] stride=1 padding=0 active=ReLU out=[6*6*128]
        分支2:
            Conv:[1*1*80] stride=1 padding=0 active=ReLU out=[6*6*80]
            Conv:[3*3*160] stride=1 padding=1 active=ReLU out=[6*6*160]
        分支3:
            Conv:[1*1*16] stride=1 padding=0 active=ReLU out=[6*6*16]
            Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[6*6*64]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[6*6*416]
            Conv:[1*1*64] stride=1 padding=1 out=[6*6*64]
        out=[6*6*(128+160+64+64)]=[6*6*416]     
    -----------------Inception 5b-------------------
    分为4支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 active=ReLU out=[6*6*192]
        分支2:
            Conv:[1*1*96] stride=1 padding=0 active=ReLU out=[6*6*96]
            Conv:[3*3*192] stride=1 padding=1 active=ReLU out=[6*6*192]
        分支3:
            Conv:[1*1*24] stride=1 padding=0 active=ReLU out=[6*6*24]
            Conv:[5*5*64] stride=1 padding=2 active=ReLU out=[6*6*64]
        分支4:
            max pooling:[3*3] stride=1 padding=1 out=[6*6*416]
            Conv:[1*1*64] stride=1 padding=1 out=[6*6*64]
        out=[7*7*(192+192+64+64)]=[6*6*512]
    -----------------layer 5-------------------
    avg pooling:
        kernel_size=[6*6] strides=1 padding=1
        out=[1*1*512]
    faltten:
        out=[512]
    FC:
        w=[512 * 52]
        out=[52]
        active=Softmax
'''


