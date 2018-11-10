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


def load_data():
    # load training data ..
    train_data = pd.read_csv(train_datafile, sep='\t', names=['X_train', 'y_train'])
    X_train, y_train = train_data['X_train'], train_data['y_train']

    # load val data ..
    val_data = pd.read_csv(val_datafile, sep='\t', names=['X_val', 'y_val'])
    X_val, y_val = val_data['X_val'], val_data['y_val']

    return (X_train, y_train), (X_val, y_val)


def load_testset():
    # load test data
    test_data = pd.read_csv(test_datafile, sep='\t', names=['X_test', 'y_test'])
    X_test, y_test = test_data['X_test'], test_data['y_test']
    return X_test, y_test


def minibatch_generator(X, y, batch_size=BATCH_SIZE):
    while True:
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            yield X_batch, y_batch


if __name__ == '__main__':
    load_data()
