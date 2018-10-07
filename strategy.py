import tensorflow as tf
import numpy as np

from data_preprocessor import DataPreprocessor
from input_helper import InputHelper
import predict

assets = [0] # assets to be considered
bar_length = 10  # Number of minutes to generate next new bar
time_steps = 18 # time steps of LSTM model

mean_std = np.load("./save/mean-std.npy")
mean, std = mean_std[0], mean_std[1]

input_helper = InputHelper()

models = {}
for asset in assets:
    models[asset] = predict.load(asset)

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

    """
    # The idea of my strategy:
    # LSTM model:
    #   1. input: open prices, increasing rates and average volumes at the previous time steps
    #   2. ouput: (probability of) the increasing rate at next time step
    """

    position_new = position_current
    
    if (counter == 0):
        memory.data_save = []
        memory.bar_save = []
    memory.data_save.append(data[:,:])
    if ((counter + 1) % bar_length == 0):
        bar = DataPreprocessor().preprocess(np.asarray(memory.data_save), bar_length)
        bar = DataPreprocessor().select_features(bar)
        shape = bar.shape
        bar = bar.reshape(shape[0], -1)
        bar = (bar - mean) / std
        memory.bar_save.append(bar.reshape(shape[1:]))
        memory.data_save.clear()
        if len(memory.bar_save) == time_steps + 1:
            for asset in assets:
                #print(np.asarray(memory.bar_save).shape)
                x, y = input_helper.generate_dataset(np.asarray(memory.bar_save), [asset], time_steps)
                pred = predict.predict(models[asset][1], models[asset][2], x, y)
                #print(pred)
                if cash_balance > 16000:
                    if pred > 0.55:
                        position_new[asset] += 1
                    elif pred > 0.7:
                        position_new[asset] += 5
                    elif pred > 0.85:
                        position_new[asset] += 10
                if position_current[asset] > 0:
                    if pred < 0.45: 
                        position_new[asset] -= 1
                    elif pred < 0.3:
                        position_new[asset] -= 5
                    elif pred < 0.15:
                        position_new[asset] = 0
            memory.bar_save.pop(0)

    # End of strategy
    return position_new, memory
