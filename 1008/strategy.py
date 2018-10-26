'''
simple rnn based method
slow 

'''
from auxiliary import generate_seg,generate_bar,generate_bar2
import pandas as pd
#from tensorflow.contrib.rnn import GRUCell as gru
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
import model_loader

bar_length = 10
time_steps=30
input_size=2
output_size = 1
num_layers=3
hidden=32
assets = [0,1,2,3]  
times = [0.1,0.05,0.1,0.1]

# pre-load model to speed up 
models = {}
for asset in assets:
    models[asset] = model_loader.load(asset)


#parameter calculated from the past data
#week 3
#upper_bound = [542.43,6799.43,249.25,60.41]
#lower_bound = [411.19,6086.15,181.75,50.79]
#week 4
#upper_bound = [471.19,6574.17,229.51,57.66]
#lower_bound = [428.83,6276.87,193.55,51.84]


#week 5
#upper_bound = [518.79,6666.12,234.18,61.63]
#lower_bound = [432.93,6396.64,210.16,55.59]

#week 6
upper_bound = [516.77,6546.40,224.39,58.64]
lower_bound = [486.58,6481.08,218.36,56.93]


#transfer the ouput
def trans_2(num,asset_index):
    r = num
    if num >= 1.8:
        return  1.025, r 
    elif num >= 0.2:
        return 1,0
    else:
        return  0.975, r 

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
    # Pattern for short signal:   
    # if next minute price is larger than the upper bounder, then short certain amount

    # Pattern for short signal:
    # if next minute price is lower than the upper bounder, then long certain amount
    
    # use rnn based model to predict the next time price 
    # input of the model is the normaled price and the volume
    # the output put of model is one value which is assumed to return 0,1,2 which stands for lower <-5%,same -5%-5%,larger >5%

    # if cash balance is lower than some value, then manually let the next position to be 0

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
            memory.pred_seq[asset_index] =0
            memory.flag[asset_index] =0
             

    
    if (counter + 1) % bar_length == 0:

        for asset_index in assets:
            memory.data_save[asset_index].loc[bar_length - 1] = data[asset_index,]
            segment = generate_seg(memory.data_save[asset_index]) # pandas dataframe
            memory.data[asset_index].append(segment) 
                
        
        if len(memory.data[asset_index]) == time_steps:
             for asset_index in assets:
                if 1:
                   
                    memory.flag[asset_index] = 1
                    memory.last[asset_index] = memory.data[asset_index][-1][0]
                    inputs = generate_bar2(memory.data[asset_index])
                    x = np.array(inputs).reshape(1,time_steps,input_size)   
                    predict = model_loader.predict(models[asset_index][1],models[asset_index][2],x)
                    
                    memory.transaction[asset_index],_ = trans_2(predict,asset_index)
                    memory.pred_seq[asset_index] = 0
                    if memory.last[asset_index] * memory.transaction[asset_index] >= upper_bound[asset_index]:
                        memory.pred_seq[asset_index] = - 1 * times[asset_index]
                    elif memory.last[asset_index] * memory.transaction[asset_index]>= lower_bound[asset_index]:
                        memory.pred_seq[asset_index] = 0
                    elif memory.last[asset_index] * memory.transaction[asset_index] < lower_bound[asset_index]:
                        memory.pred_seq[asset_index] = 1 * times[asset_index]
                    
                    if cash_balance < init_cash*0.15:
                        position_new[asset_index] = 0
                    else:
                        position_new[asset_index]  +=  memory.pred_seq[asset_index]
                   
                    
                memory.data[asset_index] = memory.data[asset_index][1:]
    else:
        for asset_index in assets:
            memory.data_save[asset_index].loc[(counter + 1) % bar_length - 1] = data[asset_index,]
                 

    # End of strategy
    return position_new, memory
