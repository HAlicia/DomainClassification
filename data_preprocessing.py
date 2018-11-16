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
import jieba
import pandas as pd


def cut2token_with_userDict(in_file, out_file, usedictList):
    # add user dict
    for userdict in usedictList:
        jieba.load_userdict(userdict)
        print("User dict {} loaded!".format(userdict))

    f = open(out_file, mode='a', encoding='utf8')
    print("Start to cut tokens ...")
    with open(in_file, 'r', encoding='utf8') as f1:
        for line in f1:
            sent = line.strip()
            token_list = jieba.cut(sent)
            new_line = " ".join(token_list)
            f.write(new_line)

def extract_new_test_data(filename, out_file, column_list, sheet_nums=[0]):
    file = os.path.join(data_dir, filename)
    out_file = os.path.join(data_dir, out_file)
    import pandas as pd
    for sheet_num in sheet_nums:
        df = pd.read_excel(file, sheet_name=sheet_num)
        df[column_list].to_csv(out_file, header=False, index=False, sep='\t', mode="a")


def read_all_sheets(in_excel, out_csv_file):
    xls = pd.ExcelFile(in_excel)
    # to read all sheets to a map
    for sheet_name in xls.sheet_names:
        print("load sheet {} ..".format(sheet_name))
        df = pd.read_excel(in_excel, sheet_name=sheet_name, usecols=[0, 1])
        df.to_csv(out_csv_file, mode='a', header=False, index=False, sep='\t')


if __name__ == '__main__':
    # in_file = os.path.join(data_dir, 'corpus.txt')
    # out_file = os.path.join(data_dir, 'corpus_cut.txt')
    # cut2token_with_userDict(in_file, out_file, [singer_dict, song_dict])

    # extract_new_test_data("raw_test_data/JAVS三期线上用户话术(未训练)测试结果-20181113.xlsx", "testSet_20181113.csv", ["话术","预期domain"])
    #
    # extract_new_test_data("raw_test_data/testSet_20181114.xlsx", out_file="testSet_20181114_cut.csv", column_list=["原始话术","Domain"])

    # extract_new_test_data("raw_test_data/贾海拥提供2_三期线上话术.xlsx", out_file="testSet_1More.csv", column_list=["话术","Domain"])

    # in_file = os.path.join(data_dir, 'testSet_20181111.csv')
    # out_file = os.path.join(data_dir, 'testSet_20181111_cut.csv')
    # cut2token_with_userDict(in_file, out_file, usedictList=[singer_dict, song_dict])

    in_excel = os.path.join(data_dir, "raw_test_data/贾海拥提供2_三期线上话术.xlsx")
    out_csv_file = os.path.join(data_dir, "1More_test.csv")
    read_all_sheets(in_excel, out_csv_file,)
