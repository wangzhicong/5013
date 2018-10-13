# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:21:15 2018

@author: Wangzhc

copy from kuojian
"""

import tensorflow as tf

def load(asset):
    
    graph = tf.Graph()
    save_name = str(asset)+'.ckpt'
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph('{}.meta'.format("./save/" + save_name))
            #saver = tf.train.Saver()
            saver.restore(sess,'save/'+save_name)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            pred = graph.get_operation_by_name("pred_val").outputs[0]
            return graph, sess, [input_x, input_y, dropout_keep_prob, pred]
            #return graph, sess 

def predict(sess, ops, x):
    prediction = sess.run(ops[3], feed_dict={ops[0]: x, ops[2]: 1.0})
    return prediction[0][0]