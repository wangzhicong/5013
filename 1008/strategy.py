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
import lstm_model
tf.logging.set_verbosity(tf.logging.ERROR)


bar_length = 1
time_steps=60
input_size=1
num_layers=3
hidden=32
assets = [0,1,2,3]
threshold = [0.1 for i in assets]




def prediction_lstm(asset_index,inputs,time_step,save_name):
    model = lstm_model.GRU_model(time_steps, input_size, num_layers, hidden)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,'save/'+save_name)
        x = np.array(inputs).reshape(1,time_step,input_size)       
        prob=sess.run(model.pred,feed_dict={model.input_x:x,model.dropout_keep_prob:0.9})
    return prob


def trans(num,asset_index):
    r = num/threshold[asset_index]
    if num >= threshold[asset_index]:
        return  1, r 
    elif num >= -threshold[asset_index]:
        return 0,0
    else:
        return  -1, r 
    
def trans_2(num,asset_index):
    r = num
    if num >= 0.7:
        return  1, r 
    elif num >= 0.3:
        return 0,0
    else:
        return  -1, r 

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
            #save_name = str(asset_index)+'.ckpt'
            memory.data_save[asset_index].loc[bar_length - 1] = data[asset_index,]
            segment = generate_seg(memory.data_save[asset_index]) # pandas dataframe
            
            memory.data[asset_index].append(segment) 
            
        #print(memory.data)     
            #inputs = generate_bar(segment)
            #tf.reset_default_graph()
            
            #predict = prediction_lstm(asset_index,inputs,interval_length,save_name)
                
        
        if len(memory.data[asset_index]) == time_steps:
             #print(counter,position_current, memory.transaction,cash_balance,total_balance) 
             for asset_index in assets:
                #print(counter,position_new) 
                
               
                
                if 1:
                    memory.flag[asset_index] = 1
                    memory.last[asset_index] = memory.data[asset_index][-1][0]
                    inpuuts = generate_bar2(memory.data[asset_index])
                    #print(asset_index,inpuuts)
                    tf.reset_default_graph()
                    save_name =str(asset_index)+'.ckpt'
                    
                    predict = prediction_lstm(asset_index,inpuuts,time_steps,save_name)
                    #print(asset_index,predict)
                    
                    # 追买追卖策略
                    memory.pred_seq[asset_index] = []
                    _,memory.transaction[asset_index] = trans(predict[-1][0],asset_index)
                    position_new[asset_index] = memory.last[asset_index] + memory.transaction[asset_index]
                    #print(memory.transaction, position_current)
                
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
                if (counter + 1) % time_steps  < time_steps* 0.3:
                    #print( value(data[asset_index,]),memory.last[asset_index])
                    if memory.transaction[asset_index] > 0:   
                        if value(data[asset_index,]) <= memory.last[asset_index] :# and position_current[asset_index]  <= memory.last[asset_index] + memory.transaction[asset_index]:
                            #print(memory.transaction[asset_index], position_current[asset_index])
                            position_new[asset_index] = memory.last[asset_index] + memory.transaction[asset_index]
                    elif memory.transaction[asset_index] < 0 :
                        #print( value(data[asset_index,]),memory.last[asset_index])
                        if value(data[asset_index,]) >= memory.last[asset_index] :# and position_current[asset_index]  <= memory.last[asset_index] + memory.transaction[asset_index]:
                            position_new[asset_index] = memory.last[asset_index] + memory.transaction[asset_index]
                    

    # End of strategy
    return position_new, memory
