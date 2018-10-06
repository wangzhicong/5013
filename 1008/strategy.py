'''
simple rnn based method
slow with low accuracy
maybe a wrong implementation
'''
from auxiliary import generate_seg,generate_bar,generate_bar2
import pandas as pd
#from tensorflow.contrib.rnn import GRUCell as gru
import tensorflow as tf
import numpy as np



tf.logging.set_verbosity(tf.logging.ERROR)
bar_length = 1 # Number of minutes to generate next new bar
interval_length = 60*3 # length for gru cell
skip_length = 1 # the skip interval for each prediction
rnn_unit= 32 # hidden layer units
input_size=2  # input size of gru cell
output_size=2 # output size of gru cell
batch_size=1 # batch_size of gru cell
time_step = interval_length

#threshold for transaction, maybe changed for different type later
threshold =[0.1 for i in range(4)]
assets = [0,1,2,3]
#transaction coef, maybe changed for different type later
coef = [1 for i in range(4)]




def trans(num,asset_index):
    r = num/threshold[asset_index]
    if num >= threshold[asset_index]:
        return  1, r * coef[asset_index]
    elif num >= -threshold[asset_index]:
        return 0,0
    else:
        return  -1, r * coef[asset_index]

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
    pred = tf.layers.dense(output,output_size)
    #w_out=weights['out']
    #b_out=biases['out']
    #pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



def prediction_lstm(asset_index,inputs,time_step,save_name):
    X=tf.placeholder(tf.float32, shape=[None,None,input_size],name='x')
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
        saver.restore(sess,'save/'+save_name)
        #print(asset_index,sess.run(weights['in']))
        #for i in range(1):
        x = np.array(inputs).reshape(1,time_step,input_size)
        
        prob=sess.run(pred,feed_dict={X:x})
            #inputs.append(prob[-1][0])
        #print(asset_index,prob)
        
        

        ##x = np.array(inputs).reshape(1,len(inputs),input_size)
        #prob=sess.run(pred,feed_dict={X:x})
        ##if trans(prob[-1][0]) * trans(prob[-len(trans)//][0]) > 0
        #predict = trans(prob[-1][0])
        #print(asset_index,prob[-1])
    return prob




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
    # Pattern for long signal:
    # When the predicted signal is larger than threshold , we long (pred/threshold) at the next bar; otherwise we short certain at the next bar.

    # Pattern for short signal:
    # When the predicted signal is smaller than -threshold , we short (pred/threshold) at the next bar;

    # No controlling of the position is conducted in this strategy.

    # Get position of last minute
    position_new = position_current.copy()


    if (counter == 0):
        memory.data_save = {}
        memory.data = {}
        memory.transaction = {}
        memory.last = {}
        memory.next = {}
        memory.flag = {}
        memory.pred_seq={}
        for asset_index in range(4):
            #memory.data_save[asset_index] = pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume'])
            memory.data_save[asset_index] = pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume'])
            memory.data[asset_index] = []
            memory.transaction[asset_index] = 0
            memory.last[asset_index] = 0
            memory.next[asset_index] = 0
            memory.pred_seq[asset_index] =[]
            memory.flag[asset_index] =0
             

    
    if (counter + 1) % bar_length == 0:
        #print(counter)
        #seg = None
        for asset_index in assets:
            save_name = str(asset_index)+'.ckpt'
            memory.data_save[asset_index].loc[bar_length - 1] = data[asset_index,]
            segment = generate_seg(memory.data_save[asset_index]) # pandas dataframe
            
            memory.data[asset_index].append(segment) 
            
        #print(memory.data)     
            #inputs = generate_bar(segment)
            #tf.reset_default_graph()
            
            #predict = prediction_lstm(asset_index,inputs,interval_length,save_name)
                
        
        if len(memory.data[asset_index]) == interval_length:
             #print(counter,position_current, memory.transaction,cash_balance,total_balance) 
             for asset_index in assets:
                #print(counter,position_new) 
                
                '''
                if memory.flag[asset_index] == 1:
                    memory.flag[asset_index] = 0
                    memory.next[asset_index] = position_current[asset_index]
                    inpuuts = generate_bar2(memory.data[asset_index])
                    tf.reset_default_graph()
                    memory.pred_seq[asset_index] = []
                    predict = prediction_lstm(asset_index,inpuuts,interval_length,save_name)
                    for i in range(int(bar_length*0.3)):
                        if predict[i][0] * memory.transaction[asset_index] < 0:
                            memory.pred_seq[asset_index].append(i)
                            
                    memory.transaction[asset_index] = memory.last[asset_index]-memory.next[asset_index]
                    position_new[asset_index] = 0 # memory.transaction[asset_index]
                    memory.last[asset_index] = position_current[asset_index]
                else:
                '''
                if 1:
                    memory.flag[asset_index] = 1
                    memory.last[asset_index] = memory.data[asset_index][-1][0]
                    inpuuts = generate_bar2(memory.data[asset_index])
                    tf.reset_default_graph()
                    #print(asset_index,inpuuts)
                    predict = prediction_lstm(asset_index,inpuuts,interval_length,save_name)
                    
                    # 追买追卖策略
                    memory.pred_seq[asset_index] = []
                    memory.transaction[asset_index],lens = trans(predict[-1][0],asset_index)
                    #print(memory.transaction[asset_index])
                    #for i in range(int(bar_length)):
                        #t,_ = trans(,asset_index)
                        #if predict[i][0] * memory.transaction[asset_index] < 0:
                       #     memory.pred_seq[asset_index].append(i)
                    
                    
                    
                    
                    
                    #position_new[asset_index] += memory.transaction[asset_index]
                    
                
                
                #memory.data_accuracy[asset_index].append([memory.data_save[asset_index].loc[bar_length - 1],predict*0.1])
                #print(position_current)
                memory.data[asset_index]= []
                
                
                
        else: 
            #print(counter)
            for asset_index in assets:
                #if memory.flag[asset_index] == 0 :
                #    if (counter + 1) % bar_length - 1 in memory.pred_seq[asset_index]:
                #        position_new[asset_index] = 0 
                #else:
                def value(data):
                    return sum(data[0:4])/4
                #print( value(data[asset_index,]),memory.last[asset_index])
                if memory.transaction[asset_index] > 0 :
                    
                    if value(data[asset_index,]) <= memory.last[asset_index] and position_current[asset_index] < 10:# and position_current[asset_index]  <= memory.last[asset_index] + memory.transaction[asset_index]:
                        position_new[asset_index] += memory.transaction[asset_index]
                elif memory.transaction[asset_index] < 0 :
                    #print( value(data[asset_index,]),memory.last[asset_index])
                    if value(data[asset_index,]) >= memory.last[asset_index] and position_current[asset_index] > -10 :# and position_current[asset_index]  <= memory.last[asset_index] + memory.transaction[asset_index]:
                        position_new[asset_index] += memory.transaction[asset_index]
                

    # End of strategy
    return position_new, memory
