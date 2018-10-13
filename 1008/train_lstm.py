# -*- coding: utf-8 -*-

import tensorflow as tf
import dataset


def init_parameters():
    #tf.app.flags.DEFINE_string("input_file", "../data/data_format2_201808.h5", "")


    tf.app.flags.DEFINE_integer("hidden", 32, "")
    tf.app.flags.DEFINE_integer("output_size", 1, "")
    tf.app.flags.DEFINE_integer("input_size", 2, "")
    tf.app.flags.DEFINE_float("lr", 1e-3, "")
    tf.app.flags.DEFINE_integer("num_layers", 3, "")
    tf.app.flags.DEFINE_integer("time_steps", 30, "")
    tf.app.flags.DEFINE_string("assets", '0,1,2,3', "")
    tf.app.flags.DEFINE_boolean("test", False, "")
    
    
    tf.app.flags.DEFINE_integer("num_epoches", 100, "")
    tf.app.flags.DEFINE_integer("batch_size", 512, "")
    
    
    tf.app.flags.DEFINE_float("dropout_keep_prob", 0.9, "")

    # Misc Parameters
    tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#init_parameters()
FLAGS = tf.app.flags.FLAGS
data_loader = dataset.dataset(FLAGS.time_steps)
data_loader.load_training('data_format1_201808.h5')
data_loader.split_data()
data_size = data_loader.data_size 

   
import warnings 
warnings.filterwarnings('ignore')
import lstm_model
        
def main(argv):
    FLAGS = tf.app.flags.FLAGS
    print("Parameters:")
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        print("{} = {}".format(attr, value))
     
    for index in [int(k) for k in FLAGS.assets.split(',')]:
        print('start training asset', index)
        tf.reset_default_graph()
        save_name =str(index)+'.ckpt'
        model = lstm_model.GRU_model(FLAGS.time_steps, FLAGS.input_size,FLAGS.output_size, FLAGS.num_layers, FLAGS.hidden)
        train_op=tf.train.AdamOptimizer(FLAGS.lr).minimize(model.loss)  
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:        
            sess.run(tf.global_variables_initializer())
            for i in range(FLAGS.num_epoches*(data_size//FLAGS.batch_size)):
                x,y = data_loader.next_batch(index,FLAGS.batch_size,FLAGS.input_size,FLAGS.output_size)
                _,loss_=sess.run([train_op,model.loss],feed_dict={model.input_x:x,model.input_y:y,model.dropout_keep_prob:0.9}) 
                
                if i % 1000 == 0: 
                    print('iter:',i,'loss:',loss_)
                    '''
                    count = 0
                    for k in range(data_loader.pred_num):
                        #model2 = lstm_model.GRU_model(FLAGS.time_steps, FLAGS.input_size, FLAGS.num_layers, FLAGS.hidden,batch_size=1)
                        x_t, y_t= data_loader.next_batch_test(index,k)
                        x_t = np.array(x_t).reshape(1,FLAGS.time_steps,FLAGS.input_size)
                        prob=sess.run(model.pred,feed_dict={model.input_x:x_t})
                        #print(prob[-1][0],y_t[0][0][-1][0])
                        if prob[-1][0] * y_t[0][0][-1] > 0:
                            count+=1
                        if prob[-1][0] == 0 and y_t[0][0][-1] == 0:
                            count+=1
                    print(count/data_loader.pred_num)
                    '''
            saver = tf.train.Saver()
            saver.save(sess, "save/"+save_name)


if __name__ == "__main__":
    tf.app.run()


