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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPool1D, Input, Concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

import os
import matplotlib
import argparse
import numpy as np

from data_loader import load_testset, load_pretrained_edmbedding, load_train_val_data

parser = argparse.ArgumentParser(description="SVM configuration ...")

# False: train, True: not train
parser.add_argument("--static_mode", choices=[True, False], default=False,
                    help="whether it is static")

parser.add_argument("--NUM_EPOCH", type=int, default=100,
                    help="NUM_EPOCH")

parser.add_argument("--WORD_EMBDEDDING_TYPE", choices=['rand', 'glove', 'fasttext', 'sg', 'cbow'], default='rand',
                    help="word embedding type")

parser.add_argument("--MAX_NUM_WORDS", type=int, default=5000,
                    help="MAX_NUM_WORDS")
parser.add_argument("--MAX_SEQUENCE_LENGTH", type=int, default=50,
                    help="MAX SEQUENCE LENGTH")
parser.add_argument("--EMBEDDING_DIM", type=int, default=100,
                    help="word embedding dimension")

parser.add_argument("--BATCH_SIZE", type=int, default=64,
                    help="BATCH_SIZE")
parser.add_argument("--filters", type=int, default=256,
                    help="filters")
parser.add_argument("--kernel_size", type=int, default=3,
                    help="kernel size")
parser.add_argument("--strides", type=int, default=1,
                    help="strides")
parser.add_argument("--padding", choices=['valid', 'same'], default='valid',
                    help="padding method")
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="hidden_dim")
parser.add_argument("--drop_prob", type=float, default=.2,
                    help="dropout_drop_prob")

parser.add_argument("--multi_channel", choices=[True, False], default=True,
                    help="use multi_channel?")

args = parser.parse_args()
static_mode = args.static_mode
EPOCH_NUM = args.NUM_EPOCH

MAX_NUM_WORDS = args.MAX_NUM_WORDS
MAX_SEQUENCE_LENGTH = args.MAX_SEQUENCE_LENGTH

BATCH_SIZE = args.BATCH_SIZE
filters = args.filters
kernel_size = args.kernel_size
strides = args.strides
padding = args.padding
hidden_dim = args.hidden_dim
drop_prob = args.drop_prob

WORD_EMBDEDDING_TYPE = args.WORD_EMBDEDDING_TYPE
EMBEDDING_DIM = args.EMBEDDING_DIM
multi_channel = args.multi_channel

if multi_channel:
    config_desc = "conv-1layer_multi-channel{}_static{}_embdeding-{}_dim{}_filters{}_kernel-size{}_strides{}_padding{}_hidden_dim{}_drop-rate{}_batch-size{}".format(
        1 if multi_channel else 0, 1 if static_mode else 0, WORD_EMBDEDDING_TYPE, EMBEDDING_DIM, filters, kernel_size,
        strides, padding, hidden_dim, drop_prob, BATCH_SIZE)
else:
    config_desc = "conv-1layer_static{}_embdeding-{}_dim{}_filters{}_kernel-size{}_strides{}_padding{}_hidden_dim{}_drop-rate{}_batch-size{}".format(
        1 if static_mode else 0, WORD_EMBDEDDING_TYPE, EMBEDDING_DIM, filters, kernel_size,
        strides, padding, hidden_dim, drop_prob, BATCH_SIZE)

print(config_desc)
print('=' * 80)

tensorboard_dir = os.path.join(log_dir, "{}_logdir".format(config_desc))
checkpoint_path = os.path.join(modelCheckpoint_dir, "{}.hdf5".format(config_desc))


class textCNN(object):
    def __init__(self, filters, kernel_size, strides, padding, hidden_dim, BATCH_SIZE, EMBEDDING_DIM, dataObj,
                 static_mode=static_mode, multi_channel=multi_channel,
                 embedding_mode=WORD_EMBDEDDING_TYPE, reduceLrOnPlateau=True, early_stopping=True):
        self.dataObj = dataObj
        self.static_mode = static_mode
        self.embedding_mode = embedding_mode
        self.reduceLrOnPlateau = reduceLrOnPlateau
        self.early_stopping = early_stopping

        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.batch_size = BATCH_SIZE
        self.hidden_dim = hidden_dim
        self.labels = dataObj.labelEncoder.classes_
        self.numClasses = len(self.labels)

        if multi_channel:
            print("Start to create multi-channel model ...")
            self.seq_input_static = Input((MAX_SEQUENCE_LENGTH,), dtype='int32')
            self.embeded_seq1 = self.set_embedding_mode(False)(self.seq_input_static)
            self.seq_input_non_static = Input((MAX_SEQUENCE_LENGTH,), dtype='int32')
            self.embeded_seq2 = self.set_embedding_mode(True)(self.seq_input_non_static)

            self.multi_channel = Concatenate(axis=-1)([self.embeded_seq1, self.embeded_seq2])
            self.dropout1 = Dropout(drop_prob)(self.multi_channel)
            self.h_cnn = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation="relu")(
                self.dropout1)
            self.h_pool = GlobalMaxPool1D()(self.h_cnn)
            self.fc1 = Dense(self.hidden_dim, activation="relu")(self.h_pool)
            self.dropout2 = Dropout(drop_prob)(self.fc1)
            self.preds = Dense(self.numClasses, activation="softmax")(self.dropout2)
            self.model = Model([self.seq_input_static, self.seq_input_non_static], self.preds)

        else:
            self.seq_input = Input((MAX_SEQUENCE_LENGTH,), dtype='int32')
            self.embeded_seq = self.set_embedding_mode(self.static_mode)(self.seq_input)
            self.dropout1 = Dropout(drop_prob)(self.embeded_seq)
            self.h_cnn = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation="relu")(
                self.dropout1)
            self.h_pool = GlobalMaxPool1D()(self.h_cnn)
            self.fc1 = Dense(self.hidden_dim, activation="relu")(self.h_pool)
            self.dropout2 = Dropout(drop_prob)(self.fc1)
            self.preds = Dense(self.numClasses, activation="softmax")(self.dropout2)
            self.model = Model(self.seq_input, self.preds)

    def train(self, X_train, y_train, checkpoint_path, tensorboard_dir):
        checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_acc", save_best_only=True, mode="max")
        tensorboard = TensorBoard(log_dir=tensorboard_dir,
                                  write_graph=True,
                                  write_grads=True,
                                  write_images=True)

        calllback_list = [checkpoint, tensorboard]
        if self.early_stopping:
            calllback_list.append(EarlyStopping(monitor="val_acc", patience=5))

        if self.reduceLrOnPlateau:
            reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-3)
            calllback_list.append(reduceLr)
        self.model.compile('adam', 'categorical_crossentropy', ['acc'])
        if multi_channel:
            self.track = self.model.fit([X_train, X_train], y_train, BATCH_SIZE, EPOCH_NUM, validation_split=.33,
                                        callbacks=calllback_list, shuffle=True)
        else:
            self.track = self.model.fit(X_train, y_train, BATCH_SIZE, EPOCH_NUM, validation_split=.33,
                                        callbacks=calllback_list, shuffle=True)

    def set_embedding_mode(self, trainable=True):

        if self.embedding_mode != 'rand':
            embedding_path = {'glove': glove100_path, 'fasttext': fasttxt100_path, 'sg': sg100_path,
                              'cbow': cbow100_path}
            embedding_file = embedding_path[WORD_EMBDEDDING_TYPE]
            print("Load {} word embedding ..".format(embedding_file))
            embdedings_index = load_pretrained_edmbedding(embedding_file)
            num_words = min(MAX_NUM_WORDS, len(self.dataObj.word_index) + 1)
            embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
            for word, i in self.dataObj.word_index.items():
                if i > MAX_NUM_WORDS: continue
                embedding_vec = embdedings_index.get(word)
                embedding_matrix[i] = embdedings_index.get(word) if embedding_vec is not None else embedding_matrix[i]
            print("{} embedding loaded!".format(self.embedding_mode))
            embed = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                              trainable=trainable)
        else:
            embed = Embedding(MAX_NUM_WORDS, self.EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=trainable)
        return embed

    def plot_fit(self, track, plot_filename=None):
        # assert len(track.history) == 4, "Error: did not fit validation data!"
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
        save_fig(plt, plot_filename=plot_filename)
        # plt.show()

    def evaluate(self, X_test, y_test, modelCheckpoint_path=None, name="test"):
        assert modelCheckpoint_path is not None, "No model checkpoint file given!"
        print("=" * 100)
        print("Start testing ...")
        print("start to load model from {} ...".format(modelCheckpoint_path))
        self.model = load_model(modelCheckpoint_path)
        if multi_channel:
            scores = self.model.evaluate([X_test,X_test], y_test, batch_size=self.batch_size, verbose=1)
        else:
            scores = self.model.evaluate(X_test, y_test, batch_size=self.batch_size, verbose=1)
        print("============== the results of {} set ==============".format(name))
        print(" Loss: {:.2f}, accuracy: {:.2f} ".format(scores[0], scores[1] * 100))

    # precision, recall
    def evaluate_with_f1_report(self, X_test, y_test, dataObj, modelCheckpoint_path=None, name="test"):
        assert modelCheckpoint_path is not None, "No model checkpoint file given!"
        print("=" * 100)
        print("Start testing ...")
        print("start to load model from {} ...".format(modelCheckpoint_path))
        try:
            self.model = load_model(modelCheckpoint_path)
        except Exception as e:
            print("exception: {}".format(e))

        X_test_pad = dataObj.process_and_pad_data(X_test, dataObj.tokenizer)
        y_test_index = dataObj.label2index(y_test)
        y_test_1hot = dataObj.label2onehot(y_test)
        if multi_channel:
            y_prob = self.model.predict([X_test_pad,X_test_pad], batch_size=self.batch_size, verbose=1)
        else:
            y_prob = self.model.predict(X_test_pad, batch_size=self.batch_size, verbose=1)
        y_pred = np.argmax(y_prob, axis=-1)
        y_true = np.argmax(y_test_1hot, axis=-1)
        target_names = []

        for i in set(np.concatenate((y_pred, y_true))):
            target_names.append(self.dataObj.labelEncoder.inverse_transform(i))

        f1_measure = classification_report(y_true, y_pred, target_names=target_names)
        # f1_measure = classification_report(y_test_index, y_pred, target_names=self.labels)
        print(f1_measure)

    def predict(self, X, modelCheckpoint_path=None):
        if modelCheckpoint_path is not None:
            self.model = load_model(modelCheckpoint_path)
        y_pred = self.model.predict(X, bactch_size=self.batch_size)
        return y_pred

    def load_model(self, modelCheckpoint_path):
        return load_model(modelCheckpoint_path)


class DataProvider(object):
    def __init__(self, MAX_NUM_WORDS=MAX_NUM_WORDS, ):
        self.X, self.y = load_train_val_data()

        # TODO: save to pickle
        self.tokenizer = Tokenizer(MAX_NUM_WORDS, char_level=True)
        self.tokenizer.fit_on_texts(self.X)
        self.word_index = self.tokenizer.word_index
        vocab_size = len(self.word_index) + 1
        print("vocab size: %s" % (vocab_size - 1))

        self.X_pad_seq = process_and_pad_data(self.X, self.tokenizer)

        self.labelEncoder = None
        self.fit_labelEncoder()
        self.y_1hot = self.label2onehot(self.y)

    def process_and_pad_data(self, X, corpus=None):
        assert self.tokenizer is not None or corpus is not None, "asssure the corpus argument or self.tokenizer exists!"
        if self.tokenizer is None:
            tokenizer = Tokenizer(MAX_NUM_WORDS)
            # if build vocab on another corpus, set corpus as not None! (give an input)
            if corpus != None:
                self.corpus = corpus
            tokenizer.fit_on_texts(self.corpus)
            word_index = tokenizer.word_index
            vocab_size = len(word_index) + 1
            print("vocab size: %s" % (vocab_size - 1))
        return pad_sequences(self.tokenizer.texts_to_sequences(X), MAX_SEQUENCE_LENGTH)

    def fit_labelEncoder(self):
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(self.y.unique())
        labels = list(self.labelEncoder.classes_)
        print("{} labels: {}".format(len(labels), labels))

    def label2index(self, y):
        # print(y.value_counts())
        return self.labelEncoder.transform(y)

    def label2onehot(self, y):
        y_le = self.label2index(y)
        return to_categorical(np.asarray(y_le).astype(np.int32))


def process_and_pad_data(X, tokenizer=None, corpus=None):
    if tokenizer is None:
        tokenizer = Tokenizer(MAX_NUM_WORDS)
        tokenizer.fit_on_texts(corpus)
        word_index = tokenizer.word_index
        vocab_size = len(word_index) + 1
        print("vocab size: %s" % (vocab_size - 1))
    return pad_sequences(tokenizer.texts_to_sequences(X), MAX_SEQUENCE_LENGTH)


def save_fig(plt, plot_filename, plot_dir="plot_dir"):
    matplotlib.use("Agg")
    print("plot_dir:", plot_dir)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    filename = os.path.join(plot_dir, plot_filename)
    plt.savefig('{}'.format(filename))
    print('{} saved!'.format(filename))


def check_tensorboard(tensorboard_dir):
    os.system("tensorboard --logdir {}".format(tensorboard_dir))


def main():
    dataObj = DataProvider(MAX_NUM_WORDS, )
    cnn = textCNN(filters, kernel_size, strides, padding, hidden_dim, BATCH_SIZE, EMBEDDING_DIM, dataObj)

    if os.path.exists(checkpoint_path):
        print("model already exist!")
        check_tensorboard(tensorboard_dir)
    else:

        cnn.train(dataObj.X_pad_seq, dataObj.y_1hot, checkpoint_path, tensorboard_dir)
        cnn.plot_fit(cnn.track, plot_filename='{}.pdf'.format(config_desc))

    X_test, y_test = load_testset()
    X_test_pad = dataObj.process_and_pad_data(X_test, dataObj.tokenizer)
    y_test_1hot = dataObj.label2onehot(y_test)
    # evaluate
    cnn.evaluate(X_test_pad, y_test_1hot, modelCheckpoint_path=checkpoint_path)


def eval_on_test_set(file=None):
    dataObj = DataProvider(MAX_NUM_WORDS, )
    if file:
        X_test, y_test = load_testset(file)
    else:
        X_test, y_test = load_testset()
    cnn = textCNN(filters, kernel_size, strides, padding, hidden_dim, BATCH_SIZE, EMBEDDING_DIM, dataObj,
                  static_mode=static_mode, embedding_mode=WORD_EMBDEDDING_TYPE)
    cnn.evaluate_with_f1_report(X_test, y_test, dataObj, modelCheckpoint_path=checkpoint_path)
    # cnn.evaluate(X_test_pad, y_test_1hot, modelCheckpoint_path=checkpoint_path)


if __name__ == '__main__':
    main()
    # ====================
    # test 1: old testset: test.csv
    # ===================
    eval_on_test_set()

    # ====================
    # test 2: testSet_20181113_cut.csv
    # ===================
    eval_on_test_set(os.path.join(data_dir, "testSet_20181113_cut.csv"))

    # ====================
    # test 3: testSet_20181114_cut.csv
    # ===================
    # eval_on_test_set(os.path.join(data_dir, "testSet_20181114_cut.csv"))

    eval_on_test_set(os.path.join(data_dir, "testSet_20181111_cut.csv"))
