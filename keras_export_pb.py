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

@file: keras_export_pb.py

@time: 19/11/2018 23:30 

@desc：       
               
'''              
from __future__ import print_function
from keras import backend as K
from keras.models import load_model

import tensorflow as tf

modelCheckpoint_path = "/Users/yekun/Documents/CODE_/DomainClassification/textCNN_keras/model _dir/conv-1layer_train1_embedding-rand_dim100_filters256_kernel-size3_strides1_paddingvalid_hidden_dim128_drop-rate0.2_batch-size64.hdf5"
model = load_model(modelCheckpoint_path)

sess = K.get_session()
frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['dense_2/Softmax'])


# input_name: input_1 (shape: [, 50])
# output_name: "dense_2/Softmax", shape: [, 20]

tf.train.write_graph(frozen_graph_def, logdir='pb_dir', name="charCNN_kernel3_filter256.pb", as_text=False)
