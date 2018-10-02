def generate_bar(data):
    ## Data is a pandas dataframe
    #import pandas as pd
    import numpy as np
    open_price = data['open'][0]
    close = data['close'][len(data) - 1]
    high = data['high'].max()
    low = data['low'].min()
    volume_ave = data['volume'].mean()
    #OHLC = pd.DataFrame(data = [[open_price, high, low, close, volume_ave]], columns = ['open', 'high', 'low', 'close', 'volume_ave'])
    output = []
    for i in range(len(data)):
        open_price = data['open'][i]
        close = data['close'][i]
        high = data['high'][i]
        low = data['low'][i]
        volume = data['volume'][i]
        t = (close - open_price)/open_price
        h = (high - open_price)/open_price
        l = (low - open_price)/open_price
        output.append([t*100,h*100,l*100])
    return output #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_ave,close],status
    
def generate_seg(data):
    open_price = data['open'][0]
    close = data['close'][len(data) - 1]
    high = data['high'].max()
    low = data['low'].min()
    volume_ave = data['volume'].mean()
    t = (close - open_price)/open_price # norm
    h = (high - open_price)/open_price # norm may use later
    l = (low - open_price)/open_price #  norm may use later
    #threshold = 0.001
    
    return [t*100]#,volume_ave]#,h*100,l*100] #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_ave,close],status
    #OHLC = pd.DataFrame(data = [[open_price, high, low, close, volume_ave]], columns = ['open', 'high', 'low', 'close', 'volume_ave'])
    
