'''
字母表

Created on 2020年12月15日

@author: luoyi
'''
from utils import LoggerFactory


logger = LoggerFactory.get_logger('root')


#    字母表
alphabet = "abdefghijklmnqrtwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
alphabet_map = {}

#    类别<->编码
def category_index(category):
    '''类别转换为编码
    '''
    if (not alphabet_map):
        i = 0
        for c in alphabet:
            alphabet_map[c] = i
            i = i + 1
            pass
        logger.info("inif category_index alphabet_map:" + str(i))
        pass
    return alphabet_map.get(category, -1)
#    编码<->类别
def index_category(index):
    '''编码转换类别
    '''
    return alphabet[index]



