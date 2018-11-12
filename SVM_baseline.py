# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @ File Name：     SVM_baseline
   @ Description :
   @ Author :       chaiyekun
   @ date：          2018/11/10
-------------------------------------------------
   @ Change Activity:
                   2018/11/11:
-------------------------------------------------
"""

from configs import *

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import argparse
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data, load_testset, split_and_load_data

parser = argparse.ArgumentParser(description="SVM configuration ...")
parser.add_argument("--feat", type=str, choices=["tfidf", "binary", "count", "freq"], default='tfidf',
                    help="the feature type to feed")
parser.add_argument("--kernel", choices=["rbf", "linear", "poly", "sigmoid"], default='rbf',
                    help="Kernel function")
parser.add_argument("--num_words", type=int, default=5000)
parser.add_argument("--max_iter", type=int, default=-1)
args = parser.parse_args()
feat = args.feat
kernel = args.kernel
num_words = args.num_words
max_iter = args.max_iter

(X_train, y_train), (X_val, y_val) = split_and_load_data(split_frac=.33)
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X_train)
print("vocab size of {} built!".format(len(tokenizer.word_index)))
le = LabelEncoder()
le.fit(y_train)
print("Start generating {} feature ... ".format(feat))
X_train_mat = tokenizer.texts_to_matrix(X_train, feat)
y_train_le = le.transform(y_train)
X_val_mat = tokenizer.texts_to_matrix(X_val, feat)
y_val_le = le.transform(y_val)
clf = svm.SVC(kernel=kernel, gamma='auto', max_iter=max_iter, decision_function_shape='ovr', verbose=True)

print("SVM model starts training ..")
clf.fit(X_train_mat, y_train_le)

print("Start evaluating ..")
score = clf.score(X_val_mat, y_val_le)

res = "SVM feat:{}, num_words: {}, val score:{}".format(feat, num_words, score)
print(res)

with open(svm_result_file, 'a') as f:
    f.write(res + '\n')

X_test, y_test = load_testset()
X_test_mat = tokenizer.texts_to_matrix(X_test, feat)
y_test_le = le.transform(y_test)
y_pred = clf.predict(X_test_mat)
confusionMat = confusion_matrix(y_test_le, y_pred)
f1_measure = classification_report(y_test_le, y_pred, target_names=list(le.classes_))

print(confusionMat)
print(f1_measure)
