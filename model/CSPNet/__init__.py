# -*- coding: utf-8 -*-  
'''
CSPNet
    思路：将输入分为两部分（默认是对半劈）
         part1过卷积运算，拿到运算后的特征图
         part2不过卷积运算，直接涂与part1运算后的结果叠加

CSPNet的主要贡献：
    增强了CNN的学习能力，能够在轻量化的同时保持准确性。
    降低计算瓶颈。
    降低内存成本。
    
CSPNet的BaseLayer网络结构：
    输入：(H, W, C)
    ---------- 分支 ----------
    part1（占比0.5）:
        过DenseBlock，拿到输出
        过Transition，降维(缩小尺寸可选)
    part2（占比0.5）: 
        不做任何操作
    ---------- 合并 ----------
    Concat: part1的输出 + part2
    Transition: 降维(缩小尺寸可选)
'''