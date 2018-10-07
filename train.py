import numpy as np
import tensorflow as tf

from input_helper import InputHelper
from lstm_network import LSTM

def init_parameters():
    tf.app.flags.DEFINE_string("input_files", "../data/data_format2_201808.h5", "")

    tf.app.flags.DEFINE_integer("bar_length", 10, "")
    tf.app.flags.DEFINE_string("assets", "0,1,2,3", "")
    tf.app.flags.DEFINE_integer("percent_dev", 10, "")
    tf.app.flags.DEFINE_boolean("regression", False, "")

    tf.app.flags.DEFINE_integer("num_epoches", 10, "")
    tf.app.flags.DEFINE_integer("batch_size", 32, "")
    tf.app.flags.DEFINE_float("learning_rate", 1e-3, "")

    tf.app.flags.DEFINE_integer("time_steps", 18, "")
    tf.app.flags.DEFINE_integer("num_layers", 3, "")
    tf.app.flags.DEFINE_integer("hidden_units", 64, "")
    tf.app.flags.DEFINE_float("l2_reg_lambda", 0.1, "")
    tf.app.flags.DEFINE_float("dropout_keep_prob", 0.6, "")

    # Misc Parameters
    tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

init_parameters()

def main(argv):
    FLAGS = tf.app.flags.FLAGS
    print("Parameters:")
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        print("{} = {}".format(attr, value))

    assets = list(map(int, FLAGS.assets.split(",")))
    input_files = FLAGS.input_files.split(",")
    input_helper = InputHelper()
    # train_set: (X_train, Y_train), where X_train's shape is [length, time_steps, len(assets) * 3], and Y_train's shape is [length, len(assets) * 3]
    train_set, dev_set = input_helper.get_dataset(input_files, FLAGS.bar_length, assets, FLAGS.time_steps, FLAGS.percent_dev, shuffle=True)
    print(train_set[0].shape, train_set[1].shape)
    print(dev_set[0].shape, dev_set[1].shape)

    # Training
    print("starting graph def")
    with tf.Graph().as_default():
        sess_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=sess_conf)
        print("session started")

        with sess.as_default():
            model = LSTM(FLAGS.batch_size, FLAGS.time_steps, len(assets), FLAGS.num_layers, FLAGS.hidden_units, FLAGS.regression)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            print("model object initialized")

        train_op = optimizer.minimize(model.loss, global_step=global_step)
        print("training operation defined")

        sess.run(tf.global_variables_initializer())
        print("variables initialized")

        def train_step(x_batch, y_batch):
            _, step, loss, accuracy = sess.run([train_op, global_step, model.loss, model.accuracy], 
                    feed_dict={model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: FLAGS.dropout_keep_prob})
            return loss, accuracy

        def dev_step(x_batch, y_batch):
            _, step, loss, accuracy = sess.run([train_op, global_step, model.loss, model.accuracy], 
                    feed_dict={model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1.0})
            return loss, accuracy
    
        num_batches_per_epoch = train_set[0].shape[0] // FLAGS.batch_size + 1 
        batches = input_helper.batch_iter(list(zip(train_set[0], train_set[1])), FLAGS.batch_size, FLAGS.num_epoches, shuffle=True)
        print("Training started")
        for _ in range(FLAGS.num_epoches * num_batches_per_epoch):
            batch = next(batches)
            x_batch, y_batch = zip(*batch)
            current_step = tf.train.global_step(sess, global_step)
            train_loss, train_accuracy = train_step(x_batch, y_batch)
            print("Step {}: training loss = {:g}, training accuracy = {:g}".format(current_step + 1, train_loss, train_accuracy))
        print("Training finished")

        dev_size = len(dev_set[0])
        dev_batches = input_helper.batch_iter(list(zip(dev_set[0], dev_set[1])), dev_size, 1, shuffle=True)
        print("Evaluating started")
        for dev_batch in dev_batches:
            x_batch, y_batch = zip(*dev_batch)
            dev_loss, dev_accuracy = dev_step(x_batch, y_batch)
            print("Testing loss = {:g}, testing accuracy = {:g}".format(dev_loss, dev_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, "./save/" + "".join(map(str, assets)) + "/" + "".join(map(str, assets)))

if __name__ == "__main__":
    tf.app.run()
