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

@time: 11/11/2018 01:28 

@desc：       
               
'''
from configs import *

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """ textCNN """

    def __init__(self, seqLen, numClasses, vocab_size, embedding_size, num_filters, filter_sizes):
        self.X = tf.placeholder(tf.int32, [None, seqLen], name="X")
        self.y = tf.placeholder(tf.float32, [None, numClasses], name='y')
        self.keep_prob = tf.placeholder(tf.float16, name="keep prob")

        with tf.name_scope("embedding"):
            W = tf.Variable(initial_value=tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0),
                            trainable=True, name='W')
            self.embeddings = tf.nn.embedding_lookup(params=W,
                                                     ids=self.X)  # NHW -> [batch_size, seqLen, embedding_size]
            self.embeddings_expanded = tf.expand_dims(input=self.embeddings,
                                                      axis=-1)  # [batch_size, seqLen, embedding_size, 1]

        pooled_output = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # conv layers
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(initial_value=tf.truncated_normal(shape=filter_shape, stddev=.1), name="conv_W")
                b = tf.Variable(initial_value=tf.constant(value=.1, shape=[num_filters]), name="conv_b")
                conv = tf.nn.conv2d(
                    input=self.embeddings_expanded, filter=W, strides=[1, 1, 1, 1], padding="VALID",
                    name="conv")  # NHWC
                h = tf.nn.relu(tf.nn.bias_add(value=conv, bias=b), name="activate fuction")
                pooled = tf.nn.max_pool(value=h, ksize=[1, seqLen - filter_size + 1, 1, 1], padding="VALID",
                                        name="max-pooling")
                pooled_output.append(pooled)
        num_filter_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(values=pooled_output, axis=-1)
        self.h_pool_flat = tf.reshape(tensor=self.h_pool, shape=[-1, num_filter_total])

        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(x=self.h_pool_flat, keep_prob=self.keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(
                initial_value=tf.truncated_normal(shape=[num_filter_total, numClasses], stddev=.1, name="W_dense"))
            b = tf.Variable(initial_value=tf.constant(.1, shape=[numClasses], name="b_dense"))
            self.out = tf.nn.xw_plus_b(x=self.h_dropout, weights=W, biases=b, name="out")
            self.predictions = tf.argmax(input=self.out, axis=1)

        with tf.name_scope("loss function"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.X, logits=self.out)
            self.loss = tf.reduce_mean(input_tensor=losses)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(x=self.predictions, y=tf.argmax(input=self.y, axis=1))
            self.accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype='float'))


def train(X_train, y_train, X_val, y_val, vocab_size, embedding_size,num_filters,filter_sizes):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model= TextCNN(seqLen=X_train.shape[1], numClasses=y_train.shape[1], vocab_size=vocab_size,
                           embedding_size=embedding_size, num_filters=num_filters, filter_sizes=filter_sizes)

            # TODO optimizer
            pass