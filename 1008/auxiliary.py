import numpy as np
def generate_bar(data):
    ## Data is a pandas dataframe
    #import pandas as pd
    #print('data',data)
    output = []
    #norm = [data[0][0],data[0][-1]]
    for i in range(len(data)):
        #print(data[i][0],data[0][-2])
        #tmp = 
        output.append(100*(sum(data[i][0:4])/sum(data[0][0:4])-1))
        
        
    return output #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_ave,close],status

def generate_bar2(data):
    output = []
    for i in range(len(data)):
        output.append(100*(data[i]/data[0]-1))
        
        
    return output #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_a

def generate_seg(data):
    #print(data)
    #open_price = data[0][-2]
    #print(data)
    open_price = data['open'][0]
    close = data['close'][len(data) - 1]
    high = data['high'].max()
    low = data['low'].min()
    volume_ave = data['volume'].mean()
    #print(data)
    close = (open_price+close+high+low)/4

      
    
    
    return close#,volume_ave]#,h*100,l*100] #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_ave,close],status
    #OHLC = pd.DataFrame(data = [[open_price, high, low, close, volume_ave]], columns = ['open', 'high', 'low', 'close', 'volume_ave'])
    
