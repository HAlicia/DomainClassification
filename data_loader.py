# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_loader
   Description :
   Author :       chaiyekun
   date：          2018/11/10
-------------------------------------------------
   Change Activity:
                   2018/11/10:
-------------------------------------------------
"""
from configs import *

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import jieba


def tokenize_jieba(data_file, outfile=None, userdictList=None):
    if userdictList != None:
        for userdict in userdictList:
            print("Loading user dict {}".format(userdict))
            jieba.load_userdict(userdict)
            print("User dict {} loaded!".format(userdict))
    else:
        print("No user dict load !!")
    df = pd.read_csv(data_file, sep='\t', names=['X', 'y'])
    df = shuffle(df)

    def series_tok(x):
        seg_list = jieba.cut(x, cut_all=False)
        return " ".join(seg_list)

    df['X_cut'] = df['X'].map(series_tok)
    df['y'] = df['y'].apply(lambda x: x.strip())

    if outfile != None:
        df[["X_cut", "y"]].to_csv(outfile, header=False, index=False, sep='\t')
        print("cut file saved to {}!".format(outfile))
    return df['X_cut'], df['y']


def load_train_val_data():
    # if os.path.exists(raw_data_cut):
    #     df = pd.read_csv(raw_data_cut, names=["X", "y"])
    #     return df["X"], df["y"].apply(lambda x: x.strip())
    # else:
    #     X, y = tokenize_jieba(data_file=raw_data, outfile=raw_data_cut, userdictList=[singer_dict, song_dict])
    #     return X, y

    # no tokenization
    df = pd.read_csv(raw_data, names=["X", "y"], sep='\t')
    return df["X"], df["y"].apply(lambda x: x.strip())


# def split_and_load_data(split_frac=.2):
#     # data = pd.read_csv(raw_data, sep='\t', names=['X', 'y'])
#     # X, y = data['X'], data['y']
#     X, y = tokenize_jieba(data_file=raw_data)
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_frac, random_state=SEED, shuffle=True)
#     return (X_train, y_train), (X_val, y_val)


def load_testset(test_file):
    # test set already cut!
    # if testset_cut is not None:
    #     df = pd.read_csv(testset_cut, names=['X', 'y'], sep='\t')
    #     return df["X"], df["y"].apply(lambda x: x.strip())
    # # load original test data
    # if os.path.exists(test_datafile_cut):
    #     df = pd.read_csv(test_datafile_cut, names=['X', 'y'])
    #     return df["X"], df["y"].apply(lambda x: x.strip())
    # else:
    #     X, y = tokenize_jieba(test_datafile, outfile=test_datafile_cut, userdictList=[singer_dict, song_dict])
    #     return X.astype(str), y.astype(str)

    df = pd.read_csv(test_file, names=['X', 'y'], sep='\t')
    return df["X"].apply(lambda x: x.strip()), df["y"].apply(lambda x: x.strip())


def load_pretrained_edmbedding(embedding_file):
    embedding_dict = {}
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word.isdigit():
                continue
            embedding_dict[word] = np.asarray(values[1:], 'float32')
    print("load {} word vectors".format(len(embedding_dict)))
    return embedding_dict


if __name__ == '__main__':
    # tokenize_jieba(data_file=raw_data)
    # load_train_val_data()
    # load_testset()
    # df = pd.read_csv(raw_data_cut, nameses=["X", "y"])
    # df.to_csv(raw_data_cut, header=False, index=False, )

    # tokenize_jieba(os.path.join(data_dir, "testSet_20181113.csv"), outfile="testSet_20181113_cut.csv",
    #                userdictList=[singer_dict, song_dict])
    # tokenize_jieba(os.path.join(data_dir, "testSet_20181114.csv"), outfile="testSet_20181114_cut.csv",
    #                userdictList=[singer_dict, song_dict])
    # tokenize_jieba(os.path.join(data_dir, "testSet_20181111.csv"), outfile="testSet_20181111_cut.csv",
    #                userdictList=[singer_dict, song_dict])

    tokenize_jieba(os.path.join(data_dir, "1More_test.csv"), outfile="1More_test_cut.csv",
                   userdictList=[singer_dict, song_dict])

    # file = "1More_test_cut_comma.csv"
    # df = pd.read_csv(file)
    # df.to_csv("1More_test_cut.csv", index=False, header=False, sep="\t")
