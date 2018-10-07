import tensorflow as tf
import numpy as np
from datetime import datetime

def load(asset):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph('{}.meta'.format("./save/" + str(asset) + "/" + str(asset)))
            saver.restore(sess, tf.train.latest_checkpoint("./save/" + str(asset)))

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            pred = graph.get_operation_by_name("pred_prob").outputs[0]
            return graph, sess, [input_x, input_y, dropout_keep_prob, pred]

def predict(sess, ops, x, y):
    prediction = sess.run(ops[3], feed_dict={ops[0]: x, ops[1]: y, ops[2]: 1.0})
    return prediction[0][0]
