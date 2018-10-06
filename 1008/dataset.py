# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:45:49 2018

@author: Wangzhc
"""

import numpy as np
import h5py
from tqdm import tqdm
from auxiliary import generate_bar
data_format1_dir = 'C:\\Users\\Wangzhc\\Desktop\\codes\\python code\\5013\\data\\'
data_format2_dir = 'C:\\Users\\Wangzhc\\Desktop\\codes\\python code\\5013\\data\\'

    

class dataset:
    def __init__(self,bar_length):
        self.bar_length = bar_length
        self.data_size = 0
        self.pred_num = 0
        self.training_set={}
        self.testing_set={}
        for i in range(4):
            self.training_set[i] = {'x':[],'y':[]}
            self.testing_set[i] = {'x':[],'y':[]}
    
    def split_data(self):
        for i in range(4):
            self.testing_set[i]['x'] = self.training_set[i]['x'][-self.pred_num:self.data_size]
            self.testing_set[i]['y'] = self.training_set[i]['y'][-self.pred_num:self.data_size]
            
            self.training_set[i]['x'] = self.training_set[i]['x'][0:self.data_size-self.pred_num]
            self.training_set[i]['y'] = self.training_set[i]['y'][0:self.data_size-self.pred_num]


        self.data_size = self.data_size - self.pred_num
    
    
    def load_training(self,filename,num = -1):
        data = {}
        for asset in range(4):
            data[asset] = []
        data_format1_path = data_format1_dir +filename
        data_format2_path = data_format2_dir + filename.replace('format1','format2')
        format1 = h5py.File(data_format1_path, mode='r')
        format2 = h5py.File(data_format2_path, mode='r')
        #assets = list(format1.keys())
        keys = list(format2.keys())
        data_load = num if num != -1 else len(keys)
    
        for i in tqdm(range(data_load)):
            for asset in range(4):
                
                if len(data[asset]) == self.bar_length-1:
                    data_cur_min = format2[keys[i]][:]
                    data[asset].append(data_cur_min[asset,])
                    segment = generate_bar(data[asset])   
                    #print(segment)
                    self.training_set[asset]['x'].append(segment[0:self.bar_length//2])
                    #tmp = []
                    '''
                    for k in range(len(segment[bar_length//2:bar_length])):
                        if segment[i] < segment[bar_length//2+i]:
                            tmp.append(2)
                    '''
                    self.training_set[asset]['y'].append(segment[self.bar_length//2:self.bar_length])
                    data[asset].pop(0)
                else:
                    data_cur_min = format2[keys[i]][:]
                    data[asset].append(data_cur_min[asset,])
            
            
        self.data_size = len(self.training_set[asset]['y'])
        self.pred_num = self.data_size* 0.2
    
    def next_batch(self,index,batch,size_in,size_out):
        x = []
        y = []
        import random
        for i in range(batch):
            rnd = random.random()
            start = int(rnd*(len(self.training_set[index]['x'])-1))
            x.append(self.training_set[index]['x'][start:start+1])
            y.append(self.training_set[index]['y'][start:start+1])
        
        x = np.array(x).reshape(batch,self.bar_length//2,size_in)
        y = np.array(y).reshape(batch,self.bar_length//2,size_out)
        #print(x)
        return x,y  
    

    def next_batch_test(self,index,start):
        x = []
        y = []
        x.append(self.training_set[index]['x'][start:start+1])
        y.append(self.training_set[index]['y'][start:start+1])
        return x , y  
    