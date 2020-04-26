# -*- coding: UTF-8 -*-
import os
__all__ = ['config', ] # import 可以导入的接口


class DictObj(object):
    # 私有变量是map
    # 设置变量的时候 初始化设置map
    def __init__(self, mp):
        self.map = mp
        # print(mp)

# set 可以省略 如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':# 初始化的设置 走默认的方法
            # print("init set attr", name ,"value:", value)
            object.__setattr__(self, name, value)
            return
        # print('set attr called ', name, value)
        self.map[name] = value
# 之所以自己新建一个类就是为了能够实现直接调用名字的功能。
    def __getattr__(self, name):
        # print('get attr called ', name)
        return  self.map[name]


config = DictObj({
    'data_root': 'D:/AIdata/dog vs cat/',
    'train_path': 'D:/AIdata/dog vs cat/train',
    'test_path': 'D:/AIdata/dog vs cat/test1',
    'csv_path': 'C:/Users/BraveY/Documents/BraveY/AI-with-code/dog-vs-cat/submission/submission_pycharm.csv',
    'tensorboard_path':'C:/Users/BraveY/Documents/BraveY/AI-with-code/dog-vs-cat/tensortboard/pycharm',
    'model_save_path':'C:/Users/BraveY/Documents/BraveY/AI-with-code/dog-vs-cat/modelDict/dogs-vs-cats_pycharm.pth'
})

def debug():
    print(config.data_root)
if __name__ == '__main__':
    debug()
    pass