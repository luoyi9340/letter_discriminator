'''
测试

Created on 2020年12月15日

@author: irenebritney
'''


from utils import LoggerFactory


logger = LoggerFactory.get_logger("dataset")
logger.info("aaaa")


log2 = LoggerFactory.get_logger("aaaa")
log2.info("aaaa")