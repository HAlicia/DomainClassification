#!/usr/bin/env python

# -*- encoding: utf-8

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

@file: textCNN.py

@time: 12/11/2018 00:46 

@desc：       
               
'''
from configs import *

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPool1D, Activation, Flatten, Input
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import numpy as np

from data_loader import load_data, load_testset, load_pretrained_edmbedding, split_and_load_data

parser = argparse.ArgumentParser(description="SVM configuration ...")

parser.add_argument("--static_mode", choices=[True, False], default=False,
                    help="whether it is static")
parser.add_argument("--USE_WORD_EMBDEDDING", choices=[True, False], default=False,
                    help="use word embedding?")
parser.add_argument("--NUM_EPOCH", type=int, default=100,
                    help="NUM_EPOCH")

# TODO
parser.add_argument("--WORD_EMBDEDDING_TYPE", choices=['glove', 'fasttext', 'sg', 'cbow'], default='glove',
                    help="word embedding type")

parser.add_argument("--MAX_NUM_WORDS", type=int, default=30000,
                    help="MAX_NUM_WORDS")
parser.add_argument("--MAX_SEQUENCE_LENGTH", type=int, default=50,
                    help="MAX SEQUENCE LENGTH")
parser.add_argument("--EMBEDDING_DIM", type=int, default=64,
                    help="word embedding dimension")

parser.add_argument("--BATCH_SIZE", type=int, default=128,
                    help="BATCH_SIZE")
parser.add_argument("--filters", type=int, default=3,
                    help="filters")
parser.add_argument("--kernel_size", type=int, default=3,
                    help="kernel size")
parser.add_argument("--strides", type=int, default=1,
                    help="strides")
parser.add_argument("--padding", choices=['valid', 'same'], default='valid',
                    help="padding method")

args = parser.parse_args()
static_mode = args.static_mode
USE_WORD_EMBDEDDING = args.USE_WORD_EMBDEDDING
EPOCH_NUM = args.NUM_EPOCH

MAX_NUM_WORDS = args.MAX_NUM_WORDS
MAX_SEQUENCE_LENGTH = args.MAX_SEQUENCE_LENGTH
EMBEDDING_DIM = args.EMBEDDING_DIM

BATCH_SIZE = args.BATCH_SIZE
filters = args.filters
kernel_size = args.kernel_size
strides = args.strides
padding = args.padding


if USE_WORD_EMBDEDDING:
    embedding_path = {'glove': '', 'fasttext': '', 'sg': '', 'cbow': ''}  # TODO
    WORD_EMBDEDDING_TYPE = args.WORD_EMBDEDDING_TYPE
    embedding_file = embedding_path[WORD_EMBDEDDING_TYPE]


class textCNN(object):
    def __init__(self, embed, filters, kernel_size, numClasses):
        self.seq_input = Input((MAX_SEQUENCE_LENGTH,), dtype='int32')
        self.embeded_seq = embed(self.seq_input)
        self.h_cnn = Conv1D(filters, kernel_size, strides, padding, activation="relu")(self.embeded_seq)
        self.h_pool = GlobalMaxPool1D()(self.h_cnn)
        self.preds = Dense(numClasses, activation="softmax")(self.h_pool)

        self.model = Model(self.seq_input, self.preds)

    def train(self, X_train, y_train, X_val, y_val):
        self.model.compile('adam', 'categorical_crossentropy', ['acc'])
        self.track = self.model.fit(X_train, y_train, BATCH_SIZE, EPOCH_NUM, validation_data=[X_val, y_val])

    def plot_fit(self, track, plot_filename=None):
        assert len(track.history) == 4, "Error: did not fit validation data!"
        acc = track.history['acc']
        val_acc = track.history['val_acc']
        loss = track.history['loss']
        val_loss = track.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'g--', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(122)
        # "bo" is for "blue dot"
        plt.plot(epochs, acc, 'b', label='Training acc')
        # b is for "solid blue line"
        plt.plot(epochs, val_acc, 'y--', label='Validation acc')
        plt.title('Training and validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.grid()
        # save_fig(plt, plot_filename=plot_filename)
        plt.show()


def data_provider():
    (X_train, y_train), (X_val, y_val) = split_and_load_data(split_frac=.33)
    tokenizer = Tokenizer(MAX_NUM_WORDS, char_level=True)
    # loc = np.where(np.isnan(np.asarray(X_train)))
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    print("vocab size: %s" % (vocab_size - 1))

    train_pad_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), MAX_SEQUENCE_LENGTH)
    val_pad_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), MAX_SEQUENCE_LENGTH)
    le = LabelEncoder()
    le.fit(y_train)
    y_train_le, y_val_le = le.transform(y_train), le.transform(y_val)
    y_train, y_val = to_categorical(np.asarray(y_train_le)), to_categorical(np.asarray(y_val_le))
    # numClasses = y_train.shape[1]

    print("X_train sequence shape:{}".format(train_pad_seq.shape))
    print("y_train shape{}".format(y_train.shape))

    global EMBEDDING_DIM

    if USE_WORD_EMBDEDDING:
        print("Load word embedding ..")
        embdedings_index = load_pretrained_edmbedding(embedding_file)  # TODO
        num_words = len(list(embdedings_index.values()))[0]
        print("Loaded {} word vectors".format(len(embdedings_index)))
        EMBEDDING_DIM = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > MAX_NUM_WORDS: continue
            embedding_vec = embdedings_index.get(word)
            embedding_matrix[i] = embdedings_index.get(word) if embedding_vec is not None else embedding_matrix[i]
        embed = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                          trainable=static_mode)
    else:
        # global EMBEDDING_DIM
        print("Start rand mode ..")
        embed = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
    return train_pad_seq, y_train, val_pad_seq, y_val, embed, tokenizer


def save_fig(plt, plot_filename, plot_dir):
    import os
    print("plot_dir:", plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    filename = os.path.join(plot_dir, plot_filename)
    plt.savefig('{}'.format(filename))
    print('{} saved!'.format(filename))


if __name__ == '__main__':
    train_pad_seq, y_train, val_pad_seq, y_val, embed, tokenizer = data_provider()
    cnn = textCNN(embed, filters, kernel_size, numClasses=y_train.shape[1])
    cnn.train(train_pad_seq, y_train, val_pad_seq, y_val)
    cnn.plot_fit(cnn.track)
