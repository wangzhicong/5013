# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:24:48 2018

@author: Wangzhc
"""

import tensorflow as tf
import numpy as np




class GRU_model:
    def __init__(self, time_steps, input_size, num_layers, hidden_units, regression=False):
        self.input_x = tf.placeholder(tf.float32, [None, time_steps, input_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, time_steps, input_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    
        self.pred = self.model(self.input_x, time_steps, input_size, num_layers, hidden_units, self.dropout_keep_prob)
        self.accuracy = tf.reduce_mean(tf.maximum(.0, tf.sign(tf.multiply(self.pred, self.input_y))))
        #if regression:
        self.loss = self.mse_loss(self.pred, self.input_y)
        #else:
        #    self.pred = tf.nn.sigmoid(self.pred, name="pred")
        #    self.loss = self.cross_entropy_loss(self.pred, self.input_y)
            
        
    def model(self, x, time_steps, input_size, num_layers, hidden_units, dropout_keep_prob):
        x = tf.unstack(x, axis=1)
        lstm_cells = []
        for _ in range(num_layers):
            lstm_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
            dropout_lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
            lstm_cells.append(dropout_lstm_cell)
        stacked_lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells, state_is_tuple=True)
        outputs, state = tf.nn.static_rnn(stacked_lstm_cells, x, dtype=tf.float32)
        outputs = tf.reshape(outputs,[-1,hidden_units])
        W = self.weights(hidden_units, input_size)
        b = self.bias(input_size)
        return tf.add(tf.matmul(outputs, W), b, name="pred")
    
    
    def mse_loss(self, pred, y):
        return tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y, [-1]))) 

    def cross_entropy_loss(self, pred, y):
        label = tf.maximum(.0, tf.sign(y))
        return tf.reduce_mean(-label * tf.log(pred) - (1 - label) * tf.log(1 - pred))

    def weights(self, input_size, output_size):
        W = tf.Variable(tf.random_normal(shape=[input_size, output_size]) * 0.01, name="weights")
        return W

    def bias(self, output_size):
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="bias")
        return b
