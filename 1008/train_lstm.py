# -*- coding: utf-8 -*-

import h5py
import sys
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np


#working_folder = 'C:\\Users\\Wangzhc\\Desktop\\codes\\python code\\5013\\Greenwich_0930\\'
#sys.path.append(working_folder)
from auxiliary import generate_seg,generate_bar

data_format1_dir = 'C:\\Users\\wangz\\Desktop\\codes\\python code\\5013\\data\\'
data_format2_dir = 'C:\\Users\\wangz\\Desktop\\codes\\python code\\5013\\data\\'


training_set={}
for i in range(4):
    training_set[i] = {'x':[],'y':[]}
    
#testing_set={'x':[],'y':[]}


filename = 'data_format1_201808.h5'
data_format2_path = data_format2_dir + filename.replace('format1','format2')
format2 = h5py.File(data_format2_path, mode='r')

keys = list(format2.keys())





bar_length = 30
time_step = bar_length //2

rnn_unit= 32       #hidden layer units
input_size=1
output_size=1
 
globalstep = 25000 # 全局下降步数 
lr = 0.01 # 初始学习率 
#decaystep = data_load * 10 # 实现衰减的频率 
decay_rate = 0.1 # 衰减率 
epoch = 100
batch_size = 64

split = 0.1
pred_num = 0 #int(data_load*split)

def next_batch(index,batch):
    x = []
    y = []
    import random
    rnd = random.random()
    #start = int(rnd*(len(training_set[index]['x'])-batch-1))
    #print(start)
    for i in range(batch):
        rnd = random.random()
        start = int(rnd*(len(training_set[index]['x'])-1))
        x.append(training_set[index]['x'][start:start+1])
        y.append(training_set[index]['y'][start:start+1])
    #print(x)
    return x,y  
    

def next_batch_test(start):
    x = []
    y = []
    for i in range(1):       
        x.append(testing_set['x'][start:start+1])
        y.append(testing_set['y'][start:start+1])
    return x , y  
    
data = {}
for asset in range(4):
    data[asset] = []
# preparing data 
def preparing_training(filename):
    data_format1_path = data_format1_dir +filename
    data_format2_path = data_format2_dir + filename.replace('format1','format2')
    format1 = h5py.File(data_format1_path, mode='r')
    format2 = h5py.File(data_format2_path, mode='r')
    #assets = list(format1.keys())
    keys = list(format2.keys())
    data_load =  len(keys)

    for i in tqdm(range(data_load)):
        for asset in range(4):
            
            if len(data[asset]) == bar_length:
                data_cur_min = format2[keys[i]][:]
                data[asset].append(data_cur_min[asset,])
                segment = generate_bar(data[asset])   
                #print(segment)
                training_set[asset]['x'].append(segment[0:bar_length//2])
                training_set[asset]['y'].append(segment[bar_length//2:bar_length])
                data[asset].pop(0)
            else:
                data_cur_min = format2[keys[i]][:]
                data[asset].append(data_cur_min[asset,])
                 
        #data.pop(0)
        #else:
            #data.append(seg)
            
            
        #x=generate_bar(datas)       
        #print(x)
        #print(x[0:len(x)-1])
        #print(x[1:len(x)])
        #training_set['x'].append(x[0:len(x)-1])
        #training_set['y'].append(x[1:len(x)])
    
    return 0

# read data from .h5 files to generate the training dataset
filename = 'data_format1_201808.h5'
preparing_training(filename)
#filename = 'data_format1_201807.h5'
#preparing_training(filename)
#filename = 'data_format1_201806.h5'
#preparing_training(filename)
#filename = 'data_format1_201805.h5'
#preparing_training(filename)
#filename = 'data_format1_20180901_20180909.h5'
#preparing_training(filename)
#filename = 'data_format1_20180909_20180916.h5'
#preparing_training(filename)
#filename = 'data_format1_20180916_20180923.h5'
#preparing_training(filename)
#filename = 'data_format1_20180923_20180930.h5'
#preparing_training(filename)
data_size = len(training_set[0]['x'])

#shift the training dataset to let the previous status to predict the future status
#training_set['x'] = training_set['x'][0:-1]
#training_set['y'] = training_set['y'][1:data_size]
#data_size-=1

#testing_set['x'] = training_set['x'][-pred_num:data_size]
#testing_set['y'] = training_set['y'][-pred_num:data_size]

#training_set['x'] = training_set['x'][0:data_size-pred_num]
#training_set['y'] = training_set['y'][0:data_size-pred_num]


data_size = data_size - pred_num


#print(len(training_set['x']),len(testing_set['x'] ))

import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')







#输入层、输出层权重、偏置

'''

def single_layer_dynamic_gru(input_size,time_step,rnn_unit,state):

    返回动态单层GRU单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数

    #可以看做隐藏层
    gru_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_unit)
    #动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    #state=gru_cell.zero_state(batch_size,dtype=tf.float32)
    hiddens,states = tf.nn.dynamic_rnn(cell=gru_cell,inputs=input_size,dtype=tf.float32)
        
    
    #注意这里输出需要转置  转换为时序优先的
    #hiddens = tf.transpose(hiddens,[1,0,2])  
    #output=tf.reshape(states,[-1,rnn_unit])
    hiddens = tf.layers.flatten(hiddens)
    
    return hiddens,states



def conv(input_data):
    network = tf.layers.conv2d(input_data,filters= 16,kernel_size=3,strides=[1,1],padding='SAME')
    #network = tf.layers.batch_normalization(network,training=True)
    network = tf.nn.relu(network)
    network = tf.layers.conv2d(network,filters= 32,kernel_size=3,strides=[1,1],padding='SAME')
    #network = tf.layers.batch_normalization(network,training=True)
    network = tf.nn.relu(network)
    network = tf.layers.flatten(network)
    return network
'''


def lstm(X,weights,biases):  
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    #cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
    #tf.nn.rnn_cell.GRUCell(num_units, input_size=None, activation=tanh)

    cell=tf.nn.rnn_cell.GRUCell(rnn_unit)
    state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=state, dtype=tf.float32,scope='Gru')  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

def train_lstm(index,save_name,time_step=time_step):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size],name='x')
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size],name='y')
    #W = tf.Variable(tf.random_normal([time_step*rnn_unit,output_size]))  #w为784*10的矩阵，输入层为784个像素点
    #B = tf.Variable(tf.random_normal([output_size]))
    #State = tf.Variable(tf.random_normal([batch_size,rnn_unit])) 
    #state = tf.placeholder(tf.float32, shape=[None,None,rnn_unit])
    #states,_=single_layer_dynamic_gru(X,time_step,rnn_unit,State)
    #states = conv(X)
    weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit]),name='in_w'),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size],name='out_w'))
         }
    biases={
                'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,]),name='in_b'),
                'out':tf.Variable(tf.constant(0.1,shape=[output_size,]),name='out_b')
                }


    pred,_ = lstm(X,weights,biases)
    
    
    #pred = tf.nn.softmax(tf.matmul(states,W)+B)
    #损失函数
    global_ = tf.Variable(tf.constant(0), trainable=False)
    #f = tf.train.natural_exp_decay(lr, global_, decaystep, decay_rate, staircase=True)   
    #l2loss = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y)) #+ 0.01*l2loss
    loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))) #+ l2loss
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)  
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #w = np.random.randn(rnn_unit,output_size)
        #b = np.random.randn(output_size)
        #states= np.random.randn(batch_size,rnn_unit)
        
        sess.run(tf.global_variables_initializer())
        #重复训练5000次
        #cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
        #state=cell.zero_state(batch_size,dtype=tf.float32)
        for i in range(epoch*(data_size//batch_size)):
            #print(i%(data_size-batch_size))
            #print(i)
            x_l,y_l = next_batch(index,batch_size)
            #print(y)
            #print(i,x,y)
            #print(x,y)
            #print(i,len(x_l[0][0][0]))
            x = np.array(x_l).reshape(batch_size,time_step,input_size)
            y = np.array(y_l).reshape(batch_size,time_step,output_size)
            _,loss_=sess.run([train_op,loss],feed_dict={X:x,Y:y,global_:i}) 
            
            #print(i,sess.run(B))
            
            #predict=[]
            if i % 1000 == 0: 
                print('iter:',i,'loss:',loss_)
                '''
                def trans(num):
                    if num >= 3:
                        return 2
                    elif num >=-3 :
                        return 1
                    else:
                        return 0
                #prev_seq=x[-1]
                #c = [0,0,0]
                #c2 = [0,0,0]
                
                for j in range(pred_num): 
                    next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
                    result = [trans(next_seq[-1][0]),trans(testing_set['x'][j][0])]
                    c[result[0]] +=1
                    c2[result[1]] +=1
                    predict.append(result) 
                    prev_seq=np.vstack((prev_seq[1:],next_seq[-1])) 
                
                count = 0
                num = 0
                for j in range(len(predict)):
                    if predict[j][0] != 1:
                        num += 1
                        if predict[j][0] == predict[j][1]:
                            count +=1
                print('iter: ',i,count,num,c,c2)
                
                
                
                #test_predict=[]
                #debug = []
                count = 0
                #cal = [0,0,0]
                #cal_2 = [0,0,0]
                for step in range(len(testing_set['x'])):
                    x,y = next_batch_test(step)
                    x = np.array(x).reshape(1,time_step,input_size)
                    #print(y)
                    prob=sess.run(pred,feed_dict={X:x})   
                    #print(prob)
                    #print(x,y,y[-1][-1][-1][0])
                    result = [trans(prob[-1][0]),trans(y[-1][-1][-1][0])]
                    c[result[0]] +=1
                    c2[result[1]] +=1
                    predict.append(result) 
                    #print(prob)
                count = 0
                num = 0
                for j in range(len(predict)):
                    if predict[j][0] != 1:
                        num += 1
                        if predict[j][0] == predict[j][1]:
                            count +=1
                print('iter: ',i,count,num,c,c2)
                
                    #print(output,y[0])
                    for i in range(batch_size):
                        tmp = prob[i][0]
                        tmp = float(tmp)
                        #output[np.argmax(tmp)]=1
                        test_predict.append([tmp,y[i][-1]])
                        if tmp <= 0.5:
                            tmp = 0
                        elif tmp<= 1.5:
                            tmp = 1
                        else:
                            tmp = 2
                        
                        #    count += 1
                        cal[tmp] +=1
                        cal_2[y[i][-1]] +=1
                '''
                #count = 0
                #for i in range(len(test_predict)):
                    #if testing_set['y'][i+time_step-1] == test_predict[i] :#and a[i] != [0,1,0] :
                        #count +=1
                #print(test_predict)      
                #print('accuracy:',count,(len(test_predict)),cal,cal_2)
                #print(state)
                
        ####predict####
        #print('final_loss',loss_)
        saver = tf.train.Saver()
        saver.save(sess, "save/"+save_name)
        
        

        '''
        test_predict = scaler_for_y.inverse_transform(test_predict)
        test_y = scaler_for_y.inverse_transform(test_y)
        rmse=np.sqrt(mean_squared_error(test_predict,test_y))
        mae = mean_absolute_error(y_pred=test_predict,y_true=test_y)
        print ('mae:',mae,'   rmse:',rmse)
        '''
    #return predict
for i in range(4):
    tf.reset_default_graph()
    save_name =str(i)+'.ckpt'
    train_lstm(i,save_name)





