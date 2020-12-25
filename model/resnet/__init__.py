'''
唯一完整训练处的模型。。。
    其他训练太慢放弃了。。。


ResNet18原生网络结构：
    输入：224 * 224 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=[7*7] stride=2 padding=3 active=relu norm=bn
        out=[112 * 112 * 64]
    max pooling:
        kernel_size=[3*3] stride=2 padding=3 
        out=[56 * 56 * 64]
    -----------------BasicBlock 1*2-------------------
    Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[3*3*64] stride=1 padding=1 norm=bn out=[56 * 56 * 64]   
    shortcut: out=[56 * 56 * 64]   
    active: relu
    times: 2（该层重复2次）
    -----------------BasicBlock 2-------------------
    Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[28 * 28 *128]
    Conv:[3*3*128] stride=1 padding=1 norm=bn out=[28 * 28 * 128]   
    shortcut: out=[28 * 28 * 128]   
    active: relu
    -----------------BasicBlock 2*1-------------------
    Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[28 * 28 *128]
    Conv:[3*3*128] stride=1 padding=1 norm=bn out=[28 * 28 * 128]   
    shortcut: out=[28 * 28 * 128]   
    active: relu
    times: 1（该层重复1次）
    -----------------BasicBlock 3-------------------
    Conv:[3*3*256] stride=2 padding=1 active=relu norm=bn out=[14 * 14 *256]
    Conv:[3*3*256] stride=1 padding=1 norm=bn out=[14 * 14 * 256]   
    shortcut: out=[14 * 14 * 256]   
    active: relu
    -----------------BasicBlock 3*1-------------------
    Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[14 * 14 *256]
    Conv:[3*3*256] stride=1 padding=1 norm=bn out=[14 * 14 * 256]   
    shortcut: out=[14 * 14 * 256]   
    active: relu
    times: 1（该层重复1次）
    -----------------BasicBlock 4-------------------
    Conv:[3*3*512] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 norm=bn out=[7 * 7 * 512]   
    shortcut: out=[7 * 7 * 512]   
    active: relu
    -----------------BasicBlock 4*1-------------------
    Conv:[3*3*512] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 norm=bn out=[7 * 7 * 512]   
    shortcut: out=[7 * 7 * 512]   
    active: relu
    times: 1（该层重复1次）
    -----------------layer 2-------------------
    Global AvgPooling: out=[1*1*512]
    FC: w=[512 * 1000] active=Softmax



ResNet34原生网络结构：
    输入：224 * 224 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=[7*7] stride=2 padding=3 active=relu norm=bn
        out=[112 * 112 * 64]
    max pooling:
        kernel_size=[3*3] stride=2 padding=3 
        out=[56 * 56 * 64]
    -----------------BasicBlock 1-------------------
    Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[3*3*64] stride=1 padding=1 norm=bn out=[56 * 56 * 64]   
    shortcut: out=[56 * 56 * 64]   
    active: relu
    -----------------BasicBlock 1*2-------------------
    Conv:[3*3] stride=1 padding=1 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[3*3] stride=1 padding=1 norm=bn out=[56 * 56 * 64]   
    shortcut: out=[56 * 56 * 64]   
    active: relu
    times: 2（该层重复2次）
    -----------------BasicBlock 2-------------------
    Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[28 * 28 *128]
    Conv:[3*3*128] stride=1 padding=1 norm=bn out=[28 * 28 * 128]   
    shortcut: out=[28 * 28 * 128]   
    active: relu
    -----------------BasicBlock 2*3-------------------
    Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[28 * 28 *128]
    Conv:[3*3*128] stride=1 padding=1 norm=bn out=[28 * 28 * 128]   
    shortcut: out=[28 * 28 * 128]   
    active: relu
    times: 3（该层重复3次）
    -----------------BasicBlock 3-------------------
    Conv:[3*3*256] stride=2 padding=1 active=relu norm=bn out=[14 * 14 *256]
    Conv:[3*3*256] stride=1 padding=1 norm=bn out=[14 * 14 * 256]   
    shortcut: out=[14 * 14 * 256]   
    active: relu
    -----------------BasicBlock 3*5-------------------
    Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[14 * 14 *256]
    Conv:[3*3*256] stride=1 padding=1 norm=bn out=[14 * 14 * 256]   
    shortcut: out=[14 * 14 * 256]   
    active: relu
    times: 5（该层重复5次）
    -----------------BasicBlock 4-------------------
    Conv:[3*3*512] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 norm=bn out=[7 * 7 * 512]   
    shortcut: out=[7 * 7 * 512]   
    active: relu
    -----------------BasicBlock 4*2-------------------
    Conv:[3*3*512] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 norm=bn out=[7 * 7 * 512]   
    shortcut: out=[7 * 7 * 512]   
    active: relu
    times: 2（该层重复2次）
    -----------------layer 2-------------------
    Global AvgPooling: out=[1*1*512]
    FC: w=[512 * 1000] active=Softmax



ResNet50原生网络
    输入：224 * 224 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=[7*7] stride=2 padding=3 active=relu norm=bn
        out=[112 * 112 * 64]
    max pooling:
        kernel_size=[3*3] stride=2 padding=3 
        out=[56 * 56 * 64]
    -----------------Bottleneck 1*3-------------------
    Conv:[1*1*64] stride=1 padding=0 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[1*1*256] stride=1 padding=0 norm=bn out=[56 * 56 * 256]
    shortcut: out=[56 * 56 * 256]   
    active: relu
    times: 3（该层重复3次）
    -----------------Bottleneck 2-------------------
    Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[56 * 56 * 128]
    Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[28 * 28 * 128]
    Conv:[1*1*512] stride=1 padding=0 norm=bn out=[28 * 28 * 512]
    shortcut: out=[28 * 28 * 512]   
    active: relu
    -----------------Bottleneck 2*3-------------------
    Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[28* 28 * 128]
    Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[28 * 28 * 128]
    Conv:[1*1*512] stride=1 padding=0 norm=bn out=[28 * 28 * 512]
    shortcut: out=[28 * 28 * 512]   
    active: relu
    times: 3（该层重复3次）
    -----------------Bottleneck 3-------------------
    Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[28 * 28 * 256]
    Conv:[3*3*256] stride=2 padding=1 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[14 * 14 * 1024]
    shortcut: out=[14 * 14 * 512]   
    active: relu
    -----------------Bottleneck 3*5-------------------
    Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[14 * 14 * 1024]
    shortcut: out=[14 * 14 * 1024]   
    active: relu
    times: 5（该层重复5次）
    -----------------Bottleneck 4-------------------
    Conv:[1*1*512] stride=1 padding=0 active=relu norm=bn out=[14 * 14 * 512]
    Conv:[3*3*512] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[1*1*2048] stride=1 padding=0 norm=bn out=[7 * 7 * 2048]
    shortcut: out=[7 * 7 * 2048]   
    active: relu
    -----------------Bottleneck 4*2-------------------
    Conv:[1*1*512] stride=1 padding=0 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[1*1*2048] stride=1 padding=0 norm=bn out=[7 * 7 * 2048]
    shortcut: out=[7 * 7 * 2048]   
    active: relu
    times: 2（该层重复2次）
    -----------------layer 2-------------------
    Global AvgPooling: out=[1*1*2048]
    FC: w=[2048 * 1000] active=Softmax
    
    
    
ResNet101原生网络
    输入：224 * 224 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=[7*7] stride=2 padding=3 active=relu norm=bn
        out=[112 * 112 * 64]
    max pooling:
        kernel_size=[3*3] stride=2 padding=3 
        out=[56 * 56 * 64]
    -----------------Bottleneck 1*3-------------------
    Conv:[1*1*64] stride=1 padding=0 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[1*1*256] stride=1 padding=0 norm=bn out=[56 * 56 * 256]
    shortcut: out=[56 * 56 * 256]   
    active: relu
    times: 3（该层重复3次）
    -----------------Bottleneck 2-------------------
    Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[56 * 56 * 128]
    Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[28 * 28 * 128]
    Conv:[1*1*512] stride=1 padding=0 norm=bn out=[28 * 28 * 512]
    shortcut: out=[28 * 28 * 512]   
    active: relu
    -----------------Bottleneck 2*3-------------------
    Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[28* 28 * 128]
    Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[28 * 28 * 128]
    Conv:[1*1*512] stride=1 padding=0 norm=bn out=[28 * 28 * 512]
    shortcut: out=[28 * 28 * 512]   
    active: relu
    times: 3（该层重复3次）
    -----------------Bottleneck 3-------------------
    Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[28 * 28 * 256]
    Conv:[3*3*256] stride=2 padding=1 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[14 * 14 * 1024]
    shortcut: out=[14 * 14 * 512]   
    active: relu
    -----------------Bottleneck 3*22-------------------
    Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[14 * 14 * 1024]
    shortcut: out=[14 * 14 * 1024]   
    active: relu
    times: 22（该层重复22次）
    -----------------Bottleneck 4-------------------
    Conv:[1*1*512] stride=1 padding=0 active=relu norm=bn out=[14 * 14 * 512]
    Conv:[3*3*512] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[1*1*2048] stride=1 padding=0 norm=bn out=[7 * 7 * 2048]
    shortcut: out=[7 * 7 * 2048]   
    active: relu
    -----------------Bottleneck 4*2-------------------
    Conv:[1*1*512] stride=1 padding=0 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[1*1*2048] stride=1 padding=0 norm=bn out=[7 * 7 * 2048]
    shortcut: out=[7 * 7 * 2048]   
    active: relu
    times: 2（该层重复2次）
    -----------------layer 2-------------------
    Global AvgPooling: out=[1*1*2048]
    FC: w=[2048 * 1000] active=Softmax    
    
    
    
ResNet152原生网络
    输入：224 * 224 * 3
    -----------------layer 1-------------------
    Conv:
        kernel_size=[7*7] stride=2 padding=3 active=relu norm=bn
        out=[112 * 112 * 64]
    max pooling:
        kernel_size=[3*3] stride=2 padding=3 
        out=[56 * 56 * 64]
    -----------------Bottleneck 1*3-------------------
    Conv:[1*1*64] stride=1 padding=0 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[3*3*64] stride=1 padding=1 active=relu norm=bn out=[56 * 56 * 64]
    Conv:[1*1*256] stride=1 padding=0 norm=bn out=[56 * 56 * 256]
    shortcut: out=[56 * 56 * 256]   
    active: relu
    times: 3（该层重复3次）
    -----------------Bottleneck 2-------------------
    Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[56 * 56 * 128]
    Conv:[3*3*128] stride=2 padding=1 active=relu norm=bn out=[28 * 28 * 128]
    Conv:[1*1*512] stride=1 padding=0 norm=bn out=[28 * 28 * 512]
    shortcut: out=[28 * 28 * 512]   
    active: relu
    -----------------Bottleneck 2*7-------------------
    Conv:[1*1*128] stride=1 padding=0 active=relu norm=bn out=[28* 28 * 128]
    Conv:[3*3*128] stride=1 padding=1 active=relu norm=bn out=[28 * 28 * 128]
    Conv:[1*1*512] stride=1 padding=0 norm=bn out=[28 * 28 * 512]
    shortcut: out=[28 * 28 * 512]   
    active: relu
    times: 7（该层重复7次）
    -----------------Bottleneck 3-------------------
    Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[28 * 28 * 256]
    Conv:[3*3*256] stride=2 padding=1 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[14 * 14 * 1024]
    shortcut: out=[14 * 14 * 512]   
    active: relu
    -----------------Bottleneck 3*35-------------------
    Conv:[1*1*256] stride=1 padding=0 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[3*3*256] stride=1 padding=1 active=relu norm=bn out=[14 * 14 * 256]
    Conv:[1*1*1024] stride=1 padding=0 norm=bn out=[14 * 14 * 1024]
    shortcut: out=[14 * 14 * 1024]   
    active: relu
    times: 35（该层重复35次）
    -----------------Bottleneck 4-------------------
    Conv:[1*1*512] stride=1 padding=0 active=relu norm=bn out=[14 * 14 * 512]
    Conv:[3*3*512] stride=2 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[1*1*2048] stride=1 padding=0 norm=bn out=[7 * 7 * 2048]
    shortcut: out=[7 * 7 * 2048]   
    active: relu
    -----------------Bottleneck 4*2-------------------
    Conv:[1*1*512] stride=1 padding=0 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[3*3*512] stride=1 padding=1 active=relu norm=bn out=[7 * 7 * 512]
    Conv:[1*1*2048] stride=1 padding=0 norm=bn out=[7 * 7 * 2048]
    shortcut: out=[7 * 7 * 2048]   
    active: relu
    times: 2（该层重复2次）
    -----------------layer 2-------------------
    Global AvgPooling: out=[1*1*2048]
    FC: w=[2048 * 1000] active=Softmax

    


'''


from model.resnet.models import ResNet_18