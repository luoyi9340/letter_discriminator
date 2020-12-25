# -*- coding: utf-8 -*-  
'''
Inception_V2模块
    结构如下：
        Inception（a）
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
                    Conv[3*3*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[3*3*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    out=M*N*C32
                - 分支4：
                    max pooling[3*3 stride=1 padding=1] out=M*N*C
                    Conv[1*1*C41 stride=1 padding=0 activation=relu] out=M*N*C41
                    out=M*N*C41
                out=[ M * N * (C11 + C22 + C32 + C41)]
            这样做的优点：
                1 两个3*3卷积核感受野等同于5*5卷积核，但参数数量小于5*5，而且增加ReLU会增加模型的非线性表述能力
        Inception（b）
            1 输入：M*N*C
            2 并行卷积核
                - 分支1：
                    Conv[1*1*C11 1stride=1 padding=0 activation=relu] out=M*N*C11
                    out=M*N*C11
                - 分支2：
                    Conv[1*1*C21 stride=1 padding=0 activation=relu] out=M*N*C21
                    Conv[n*1*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                    Conv[1*n*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                    out=M*N*C22
                - 分支3：
                    Conv[1*1*C31 stride=1 padding=0 activation=relu] out=M*N*C31
                    Conv[n*1*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[1*n*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[n*1*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[1*n*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    out=M*N*C32
                - 分支4：
                    max pooling[3*3 stride=1 padding=1] out=M*N*C
                    Conv[1*1*C41 stride=1 padding=0 activation=relu] out=M*N*C41
                    out=M*N*C41
                out=[ M * N * (C11 + C22 + C32 + C41)]           
            这样做的优点：
                1 使用n*1和1*n代替n*n卷积核，进一步减小参数。但有使用限制（见设计原则5）
        Inception（b）
            1 输入：M*N*C
            2 并行卷积核
                - 分支1：
                    Conv[1*1*C11 1stride=1 padding=0 activation=relu] out=M*N*C11
                    out=M*N*C11
                - 分支2：
                    Conv[1*1*C21 stride=1 padding=0 activation=relu] out=M*N*C21
                    Conv[n*1*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                    Conv[1*n*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                    out=M*N*C22
                - 分支3：
                    Conv[1*1*C31 stride=1 padding=0 activation=relu] out=M*N*C31
                    Conv[n*1*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[1*n*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[n*1*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[1*n*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    out=M*N*C32
                - 分支4：
                    max pooling[3*3 stride=1 padding=1] out=M*N*C
                    Conv[1*1*C41 stride=1 padding=0 activation=relu] out=M*N*C41
                    out=M*N*C41
                out=[ M * N * (C11 + C22 + C32 + C41)]           
            这样做的优点：
                1 使用n*1和1*n代替n*n卷积核，进一步减小参数。但有使用限制（见设计原则5）
        Inception（c）
            1 输入：M*N*C
            2 并行卷积核
                - 分支1：
                    Conv[1*1*C11 1stride=1 padding=0 activation=relu] out=M*N*C11
                    out=M*N*C11
                - 分支2：
                    Conv[1*1*C21 stride=1 padding=0 activation=relu] out=M*N*C21
                    Conv[n*1*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                    Conv[1*n*C22 stride=1 padding=0 activation=relu] out=M*N*C22
                    out=M*N*C22
                - 分支3：
                    Conv[1*1*C31 stride=1 padding=0 activation=relu] out=M*N*C31
                    Conv[n*1*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[1*n*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[n*1*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    Conv[1*n*C32 stride=1 padding=0 activation=relu] out=M*N*C32
                    out=M*N*C32
                - 分支4：
                    max pooling[3*3 stride=1 padding=1] out=M*N*C
                    Conv[1*1*C41 stride=1 padding=0 activation=relu] out=M*N*C41
                    out=M*N*C41
                out=[ M * N * (C11 + C22 + C32 + C41)]           
            这样做的优点：
                1 使用n*1和1*n代替n*n卷积核，进一步减小参数。但有使用限制（见设计原则5）
        
       
       
GoogleLeNet_V2原生网络（其实是V3）
    输入：299 * 299 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=[3*3*32] stride=2 padding=0 norm=bn active=relu
        out=149*149*32
    Conv:
        kernel_size=[3*3*32] stride=1 padding=0 norm=bn active=relu
        out=147*147*32
    Conv:
        kernel_size=[3*3*64] stride=1 padding=1 norm=bn active=relu
        out=147*147*64
    max pooling:
        kernel_size=[3*3] stride=2 pading=0
        out=73*73*64
    -----------------layer 2-------------------
    Conv:
        kernel_size=[3*3*80] stride=1 padding=0 norm=bn active=relu 
        out=71*71*80
    Conv:
        kernel_size=[3*3*192] stride=1 padding=1 norm=bn active=relu 
        out=71*71*192
    max pooling:
        kernel_size=[3*3] stride=2 pading=0 
        out=35*35*192
    -----------------Inception 3a-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[35*35*48]
            Conv:[5*5*64] stride=1 padding=2 norm=BN active=ReLU out=[35*35*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[35*35*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[35*35*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[35*35*192]
            Conv:[1*1*32] stride=1 padding=0 out=[35*35*32]
        out=[35*35*(64+64+96+32)]=[35*35*256]
    -----------------Inception 3b-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[35*35*48]
            Conv:[5*5*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[35*35*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[35*35*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[35*35*256]
            Conv:[1*1*64] stride=1 padding=1 out=[35*35*64]
        out=[35*35*(64+64+96+64)]=[35*35*288]
    -----------------Inception 3c-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[35*35*48]
            Conv:[5*5*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[35*35*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[35*35*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[35*35*288]
            Conv:[1*1*64] stride=1 padding=1 out=[35*35*64]
        out=[35*35*(64+64+96+64)]=[35*35*288]
    -----------------Inception 4a-------------------    
    分为3支：
        分支1:
            Conv:[3*3*384] stride=2 padding=0 norm=BN active=ReLU out=[17*17*384]
        分支2:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[35*35*64]
            Conv:[3*3*96] stride=2 padding=0 norm=BN active=ReLU out=[17*17*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[17*17*96]
        分支3:
            max pooling:[3*3] stride=2 padding=0 out=[17*17*288]
        out=[17*17*(384+96+288)]=[17*17*768]
    -----------------Inception 4b-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        分支2:
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[17*17*128]
            Conv:[1*7*128] stride=1 padding=3 norm=BN active=ReLU out=[17*17*128]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支3:
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[17*17*128]
            Conv:[7*1*128] stride=1 padding=3 norm=BN active=ReLU out=[17*17*128]
            Conv:[1*7*128] stride=1 padding=3 norm=BN active=ReLU out=[17*17*128]
            Conv:[7*1*128] stride=1 padding=3 norm=BN active=ReLU out=[17*17*128]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[17*17*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        out=[17*17*(192+192+192+192)]=[17*17*768]
    -----------------Inception 4c-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[17*17*160]
            Conv:[1*7*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支3:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[17*17*160]
            Conv:[7*1*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[1*7*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[7*1*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[17*17*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        out=[17*17*(192+192+192+192)]=[17*17*768]
    -----------------Inception 4d-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[17*17*160]
            Conv:[1*7*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支3:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[17*17*160]
            Conv:[7*1*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[1*7*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[7*1*160] stride=1 padding=3 norm=BN active=ReLU out=[17*17*160]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[17*17*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        out=[17*17*(192+192+192+192)]=[17*17*768]
    -----------------Inception 4e-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支3:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[17*17*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
        out=[17*17*(192+192+192+192)]=[17*17*768]
        
        out外接辅助分类器（作者后来自己也说，辅助分类器其实并没有起到加速收敛的作用。。。）：
            avg pooling:[5*5] stride=3 padding=0 out=[5*5*768]
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[5*5*128]
            Conv:[5*5*1024] stride=1 padding=0 norm=BN active=ReLU out=[1*1*1024]
            Fatten:out=[1024]
            FC:w=[1024 * 1000] active=Softmax out=[1000]
    -----------------Inception 5a-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
            Conv:[3*3*320] stride=2 padding=0 norm=BN active=ReLU out=[8*8*320]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[17*17*192]
            Conv:[1*7*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
            Conv:[7*1*192] stride=1 padding=3 norm=BN active=ReLU out=[17*17*192]
            Conv:[3*3*192] stride=2 padding=0 norm=BN active=ReLU out=[8*8*192]
        分支3:
            avg pooling:[3*3] stride=2 padding=0 out=[8*8*768]
        out=[8*8*(320+192+768)]=[8*8*1280]
    -----------------Inception 5b-------------------    
    分为3支：
        分支1:
            Conv:[1*1*320] stride=1 padding=0 norm=BN active=ReLU out=[8*8*320]
        分支2:
            Conv:[1*1*384] stride=1 padding=0 norm=BN active=ReLU out=[8*8*384]
            分为2支：
                分支1：Conv:[1*3*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                分支2：Conv:[3*1*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                out=[8*8*(384+384)]=[8*8*768]
        分支3:
            Conv:[1*1*448] stride=1 padding=0 norm=BN active=ReLU out=[8*8*448]
            Conv:[3*3*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
            分为2支：
                分支1：Conv:[1*3*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                分支2：Conv:[3*1*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                out=[8*8*(384+384)]=[8*8*768]
        分支4：    
            avg pooling:[3*3] stride=1 padding=1 out=[8*8*1280]
            Conv:[1*1*192] stride=1 padding=1 norm=BN active=ReLU out=[8*8*192]
        out=[8*8*(320+768+768+192)]=[8*8*2048]
    -----------------Inception 5c-------------------    
    分为3支：
        分支1:
            Conv:[1*1*320] stride=1 padding=0 norm=BN active=ReLU out=[8*8*320]
        分支2:
            Conv:[1*1*384] stride=1 padding=0 norm=BN active=ReLU out=[8*8*384]
            分为2支：
                分支1：Conv:[1*3*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                分支2：Conv:[3*1*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                out=[8*8*(384+384)]=[8*8*768]
        分支3:
            Conv:[1*1*448] stride=1 padding=0 norm=BN active=ReLU out=[8*8*448]
            Conv:[3*3*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
            分为2支：
                分支1：Conv:[1*3*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                分支2：Conv:[3*1*384] stride=1 padding=1 norm=BN active=ReLU out=[8*8*384]
                out=[8*8*(384+384)]=[8*8*768]
        分支4：    
            avg pooling:[3*3] stride=1 padding=1 out=[8*8*1280]
            Conv:[1*1*192] stride=1 padding=1 norm=BN active=ReLU out=[8*8*192]
        out=[8*8*(320+768+768+192)]=[8*8*2048]
    -----------------layer 3-------------------
    avg pool:
        kernel_size=[8*8] stride=1 padding=0 
        out=[1*1*2048]
    dropout:
        0.5
    Conv:
        kernel_size=[1*1*1000] stride=1 paddong=0 norm=BN active=ReLU
        out=[1*1*1000]
    Fatten:
        out=[1000]
    Softmax:
        out=[1000]个分类的概率
        

GoogleLeNet_V2简化版网络
    输入：100 * 100 * 1
    -----------------layer 1-------------------
    Conv:
        kernel_size=[3*3*32] stride=1 padding=0 norm=bn active=relu
        out=98*98*32
    Conv:
        kernel_size=[3*3*32] stride=1 padding=0 norm=bn active=relu
        out=96*96*32
    Conv:
        kernel_size=[3*3*64] stride=1 padding=1 norm=bn active=relu
        out=96*96*64
    max pooling:
        kernel_size=[2*2] stride=2 pading=0
        out=48*48*64
    -----------------layer 2-------------------
    Conv:
        kernel_size=[3*3*80] stride=1 padding=0 norm=bn active=relu 
        out=45*45*80
    Conv:
        kernel_size=[3*3*192] stride=1 padding=1 norm=bn active=relu 
        out=45*45*192
    max pooling:
        kernel_size=[3*3] stride=2 pading=0 
        out=22*22*192
    -----------------Inception 3a-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[22*22*48]
            Conv:[5*5*64] stride=1 padding=2 norm=BN active=ReLU out=[22*22*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[22*22*192]
            Conv:[1*1*32] stride=1 padding=0 out=[22*22*32]
        out=[22*22*(64+64+96+32)]=[22*22*256]
    -----------------Inception 3b-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[22*22*48]
            Conv:[5*5*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[22*22*256]
            Conv:[1*1*64] stride=1 padding=1 out=[22*22*64]
        out=[22*22*(64+64+96+64)]=[22*22*288]
    -----------------Inception 3c-------------------
    分为4支：
        分支1:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支2:
            Conv:[1*1*48] stride=1 padding=0 norm=BN active=ReLU out=[22*22*48]
            Conv:[5*5*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
        分支3:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[22*22*96]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[22*22*288]
            Conv:[1*1*64] stride=1 padding=1 out=[22*22*64]
        out=[22*22*(64+64+96+64)]=[22*22*288]
    -----------------Inception 4a-------------------    
    分为3支：
        分支1:
            Conv:[3*3*384] stride=2 padding=0 norm=BN active=ReLU out=[10*10*384]
        分支2:
            Conv:[1*1*64] stride=1 padding=0 norm=BN active=ReLU out=[22*22*64]
            Conv:[3*3*96] stride=2 padding=0 norm=BN active=ReLU out=[10*10*96]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[10*10*96]
        分支3:
            max pooling:[3*3] stride=2 padding=0 out=[10*10*288]
        out=[10*10*(384+96+288)]=[10*10*768]
    -----------------Inception 4b-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[10*10*128]
            Conv:[1*5*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[10*10*128]
            Conv:[5*1*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[1*5*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[5*1*128] stride=1 padding=3 norm=BN active=ReLU out=[10*10*128]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
    -----------------Inception 4c-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
    -----------------Inception 4d-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[5*1*160] stride=1 padding=3 norm=BN active=ReLU out=[10*10*160]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
    -----------------Inception 4e-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支3:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
        分支4:
            avg pooling:[3*3] stride=1 padding=1 out=[10*10*768]
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
        out=[10*10*(192+192+192+192)]=[10*10*768]
        
        out外接辅助分类器（作者后来自己也说，辅助分类器其实并没有起到加速收敛的作用。。。）：
            avg pooling:[5*5] stride=3 padding=0 out=[5*5*768]
            Conv:[1*1*128] stride=1 padding=0 norm=BN active=ReLU out=[5*5*128]
            Conv:[5*5*1024] stride=1 padding=0 norm=BN active=ReLU out=[1*1*1024]
            Fatten:out=[1024]
            FC:w=[1024 * 1000] active=Softmax out=[1000]
    -----------------Inception 5a-------------------    
    分为3支：
        分支1:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[3*3*320] stride=2 padding=0 norm=BN active=ReLU out=[4*4*320]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[10*10*192]
            Conv:[1*5*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[5*1*192] stride=1 padding=3 norm=BN active=ReLU out=[10*10*192]
            Conv:[3*3*192] stride=2 padding=0 norm=BN active=ReLU out=[4*4*192]
        分支3:
            avg pooling:[3*3] stride=2 padding=0 out=[4*4*768]
        out=[4*4*(320+192+768)]=[4*4*1280]
    -----------------Inception 5b-------------------    
    分为3支：
        分支1:
            Conv:[1*1*160] stride=1 padding=0 norm=BN active=ReLU out=[4*4*160]
        分支2:
            Conv:[1*1*192] stride=1 padding=0 norm=BN active=ReLU out=[4*4*192]
            分为2支：
                分支1：Conv:[1*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                分支2：Conv:[3*1*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                out=[4*4*(192+192)]=[4*4*384]
        分支3:
            Conv:[1*1*224] stride=1 padding=0 norm=BN active=ReLU out=[4*4*224]
            Conv:[3*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
            分为2支：
                分支1：Conv:[1*3*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                分支2：Conv:[3*1*192] stride=1 padding=1 norm=BN active=ReLU out=[4*4*192]
                out=[4*4*(192+192)]=[4*4*384]
        分支4：    
            avg pooling:[3*3] stride=1 padding=1 out=[4*4*1280]
            Conv:[1*1*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
        out=[4*4*(160+384+384+96)]=[4*4*1024]
    -----------------Inception 5c-------------------    
    分为3支：
        分支1:
            Conv:[1*1*80] stride=1 padding=0 norm=BN active=ReLU out=[4*4*80]
        分支2:
            Conv:[1*1*96] stride=1 padding=0 norm=BN active=ReLU out=[4*4*96]
            分为2支：
                分支1：Conv:[1*3*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                分支2：Conv:[3*1*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                out=[4*4*(96+96)]=[4*4*192]
        分支3:
            Conv:[1*1*112] stride=1 padding=0 norm=BN active=ReLU out=[4*4*112]
            Conv:[3*3*96] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
            分为2支：
                分支1：Conv:[1*3*384] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                分支2：Conv:[3*1*384] stride=1 padding=1 norm=BN active=ReLU out=[4*4*96]
                out=[4*4*(96+96)]=[4*4*192]
        分支4：    
            avg pooling:[3*3] stride=1 padding=1 out=[4*4*2048]
            Conv:[1*1*48] stride=1 padding=1 norm=BN active=ReLU out=[4*4*48]
        out=[4*4*(80+192+192+48)]=[4*4*512]
    -----------------layer 3-------------------
    avg pool:
        kernel_size=[4*4] stride=1 padding=0 
        out=[1*1*512]
    Conv:
        kernel_size=[1*1*52] stride=1 padding=0 norm=BN active=ReLU
        out=[1*1*52]
    Fatten:
        out=[52]
    Softmax:
        out=[52]个分类的概率
'''