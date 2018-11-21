#!/usr/bin/env python

#-*- encoding: utf-8 

'''
                      ______   ___  __
                     / ___\ \ / / |/ /
                    | |    \ V /| ' / 
                    | |___  | | | . \ 
                     \____| |_| |_|\_\
 ==========================================================================
@author: Yekun Chai

@license: School of Informatics, Edinburgh

@contact: chaiyekun@gmail.com

@file: process_data.py

@time: 21/11/2018 10:00 

@desc：       
               
'''


import pandas as pd


def merge_data(file_list, out_file):
    f = open(out_file, 'w')
    for file in file_list:
        with open(file) as f1:
            for line in f1:
                f.write(line)
        f.write('\n')
    f.close()

if __name__ == '__main__':
    file_list = ["data/train_data_10w.csv",
                 "data/trainSet_20181111.csv",
                 ]
    outfile = 'data/all_data.csv'
    # merge_data(file_list, outfile)

    # ================
    # shuffle
    # ================
    shuffle_file = 'data/all_train_shuffle.csv'
    df = pd.read_csv(outfile, sep='\t', names=['X','y'])
    # df.sample(frac=1)

    from sklearn.utils import shuffle
    df = shuffle(df)
    df.to_csv(shuffle_file, sep='\t', index=False, header=False)
    # ================

    # res_file = "/Users/yekun/Documents/CODE_/DomainClassification/textCNN_keras/181120_合并模型测试结果_26.xlsx"
    # df = pd.read_excel(res_file, sheet_name=0, usecols=[0,4])
    # df.to_csv('realTest.csv', sep='\t', header=False, index=False)