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
        self.input_y = tf.placeholder(tf.float32, [None, input_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #self.batch_size = batch_size
        self.pred = self.model(self.input_x, time_steps, input_size, num_layers, hidden_units, self.dropout_keep_prob)
        #self.accuracy = tf.reduce_mean(tf.maximum(.0, tf.sign(tf.multiply(self.pred, self.input_y))))
        #if regression:
        self.loss = self.mse_loss(self.pred,self.input_y)
        #else:
        #    self.pred = tf.nn.sigmoid(self.pred, name="pred")
        #    self.loss = self.cross_entropy_loss(self.pred, self.input_y)
            
        
    def model(self, x, time_steps, input_size, num_layers, hidden_units, dropout_keep_prob):
        #x = tf.unstack(x, axis=1)
        batch_size=tf.shape(x)[0]
        W_in = self.weights(input_size,hidden_units)
        b_in = self.bias(hidden_units)
        x = tf.reshape(x,[-1,input_size])
        x = tf.matmul(x, W_in) + b_in
        x = tf.reshape(x,[-1,time_steps,hidden_units])
        #lstm_cells = []
        #for _ in range(num_layers):
        lstm_cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, x , initial_state=state,dtype=tf.float32)
        outputs = tf.reshape(state,[-1,hidden_units])
        W = self.weights(hidden_units, input_size)
        b = self.bias(input_size)
        return tf.matmul(outputs, W)+b
    
    
    def mse_loss(self, pred, y):
        return tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y, [-1]))) 
        #return tf.reduce_mean(tf.square(tf.subtract(pred, y)))

    def cross_entropy_loss(self, pred, y):
        label = tf.maximum(.0, tf.sign(y))
        return tf.reduce_mean(-label * tf.log(pred) - (1 - label) * tf.log(1 - pred))

    def weights(self, input_size, output_size):
        W = tf.Variable(tf.random_normal(shape=[input_size, output_size]) * 0.01, name="weights")
        return W

    def bias(self, output_size):
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="bias")
        return b
    def load(self,model):
        pass
