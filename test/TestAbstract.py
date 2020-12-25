# -*- coding: utf-8 -*-  
'''
Created on 2020年12月16日

@author: irenebritney
'''
import abc


#    metaclass=abc.ABCMeta定义AClass为抽象类
class AClass(metaclass=abc.ABCMeta):
    def __init__(self, a="a"):
        self._a = a
        pass
    
    
    #    定义抽象方法
    @abc.abstractclassmethod
    def m1(self):
        pass
    
    pass
    
    
class SubClass1(AClass):
    def __init__(self, a="a"):
        super(SubClass1, self).__init__(a)
        pass
    def m1(self):
        print("in SubClass1 a:" + self._a)
        pass
    pass


a = (1, 1, 52)
b = (None, 1, 1, 52)
b = tuple(filter(None, b))
print(a == b)


