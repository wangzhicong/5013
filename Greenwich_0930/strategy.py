'''
simple rnn based method
slow with low accuracy
maybe a wrong implementation
'''
from auxiliary import generate_seg
import pandas as pd
#from tensorflow.contrib.rnn import GRUCell as gru
import tensorflow as tf
import numpy as np



tf.logging.set_verbosity(tf.logging.ERROR)
bar_length = 15  # Number of minutes to generate next new bar
interval_length = 30 # length for gru cell
skip_length = 1 # the skip interval for each prediction
rnn_unit= 128 # hidden layer units
input_size=4  # input size of gru cell
output_size=4 # output size of gru cell
batch_size=1 # batch_size of gru cell
time_step = interval_length

#threshold for transaction, maybe changed for different type later
threshold =[0.5 for i in range(4)]
assets = [0,1,2,3]
#transaction coef, maybe changed for different type later
coef = [1 for i in range(4)]

def lstm(X,weights,biases):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    X=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(X,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.GRUCell(rnn_unit)
    state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=state, dtype=tf.float32,scope='Gru')  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



def prediction_lstm(inputs,time_step,save_name):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size],name='x')
    weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit]),name='in_w'),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size],name='out_w'))
         }
    biases={
                'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,]),name='in_b'),
                'out':tf.Variable(tf.constant(0.1,shape=[output_size,]),name='out_b')
                }
    pred,_ = lstm(X,weights,biases)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,'../save/'+save_name)
        #print(asset_index,sess.run(weights['in']))
        def trans(num,asset_index):
            r = -num/threshold[asset_index]
            if num >= threshold[asset_index]:
                return   r * coef[asset_index]
            elif num >= -threshold[asset_index]:
                return 0
            else:
                return  r * coef[asset_index]

        x = np.array(inputs).reshape(1,time_step,input_size)
        prob=sess.run(pred,feed_dict={X:x})
        predict = []
        for asset_index in assets:
            predict.append(trans(prob[-1][asset_index],asset_index))
        #print(prob[-1])
    return predict




#model = joblib.load('model.pkl')#####
#asset_index = 1  # only consider BTC (the **second** crypto currency in dataset)

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use


def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    # The idea of my strategy:
    # simple GRU model, use (in-out)/in as inputs and prediction

    # Pattern for long signal:
    # When the predicted signal is larger than threshold , we long (pred/threshold) at the next bar; otherwise we short certain at the next bar.

    # Pattern for short signal:
    # When the predicted signal is smaller than -threshold , we short (pred/threshold) at the next bar;

    # No controlling of the position is conducted in this strategy.

    # Get position of last minute
    position_new = position_current


    if (counter == 0):
        memory.data_save = {}
        memory.data = []
        for asset_index in range(4):
            memory.data_save[asset_index] = pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume'])
            #memory.data[asset_index] = []
            memory.step = 1.02

    save_name = '1.ckpt'
    if ((counter + 1) % bar_length == 0):
        tmp = []
        for asset_index in range(4):
            memory.data_save[asset_index].loc[bar_length - 1] = data[asset_index,]
            segment = generate_seg(memory.data_save[asset_index]) # pandas dataframe
            tmp += segment
        memory.data.append(tmp)
        if len(memory.data) == interval_length:
            tf.reset_default_graph()
            #print(len(memory.data),len(memory.data[0]))
            predict = prediction_lstm(memory.data,interval_length,save_name)
            #print(predict)


            for asset_index in assets:
                if abs(position_current[asset_index]) > 20:
                    if  position_new[asset_index] * predict[asset_index] < 0 and cash_balance > init_cash *  0.2:
                        position_new[asset_index] += predict[asset_index]
                else:
                    if cash_balance < init_cash *  0.2: # avoid stop strategy                    
                        if predict[asset_index] < 0: # only selling is allowed
                            position_new[asset_index] += predict[asset_index]
                    else:
    
                        position_new[asset_index] +=  predict[asset_index]
               
            print(position_new,cash_balance,total_balance)
            memory.data = memory.data[skip_length:len(memory.data)].copy()
    else:
        for asset_index in range(4):
            memory.data_save[asset_index].loc[(counter + 1) % bar_length - 1] = data[asset_index,]

    '''
    if total_balance >= memory.step * init_cash:
         for asset_index in assets:
            position_new[asset_index] -= 1
         memory.step += 0.02
    '''


    '''
    index = -1
    if  pred != []:
        abspred = [abs(i) for i in pred]
        index = np.argmax(abspred)
        print(pred,index,abspred)

        if cash_balance < init_cash*0.2: # avoid stop strategy
            if predict < 0: # only selling is allowed
                position_new[index] += pred[index] / threshold[index]
        else:
            position_new[index] += pred[index] / threshold[index]
    '''

    # End of strategy
    return position_new, memory
