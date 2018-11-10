# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_preprocessing
   Description :
   Author :       chaiyekun
   date：          2018/11/10
-------------------------------------------------
   Change Activity:
                   2018/11/10:
-------------------------------------------------
"""
#
from configs import *
def demo_data_process(filename, new_filename = 'train.csv', cnt=7000):

    f1 = open(new_filename, 'a', encoding='utf8')
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line_X = line.split('\t')[0]
            if line_X.count(' ')>3:
                cnt -= 1
                f1.write(line)
            if cnt == 0:
                break
    f1.close()
    print(new_filename, "generated!")


if __name__ == '__main__':
    demo_data_process(raw_data, train_datafile, 7000)
    demo_data_process(raw_data, val_datafile, 1500)
    demo_data_process(raw_data, test_datafile, 1500)