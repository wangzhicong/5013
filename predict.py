import tensorflow as tf
import numpy as np

def predict(asset, x, y):
    FLAGS = tf.app.flags.FLAGS
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph('{}.meta'.format("./save/" + str(asset) + "/" + str(asset)))
            saver.restore(sess, tf.train.latest_checkpoint("./save/" + str(asset)))

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            
            pred = graph.get_operation_by_name("pred").outputs[0]

            prediction = sess.run(pred, feed_dict={input_x: x, input_y: y, dropout_keep_prob: 1.0})

            return prediction

