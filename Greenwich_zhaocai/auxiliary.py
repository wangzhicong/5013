import pandas as pd
import numpy as np
def generate_timeseries(prices, n):
    """

    Args:
        prices: A numpy array of floats representing prices
        n: An integer 720 representing the length of time series.

    Returns:
        A 2-dimensional numpy array of size (len(prices)-n) x (n+1). Each row
        represents a time series of length n and its corresponding label
        (n+1-th column).
    """
    m = len(prices) - n+1
    ts = np.empty((m, n ))
    for i in range(m):
        ts[i, :n] = prices[i:i + n]

    return ts

def generate_series(data, bar_length):
    '''
    return a DataFrame
    :param data:
    :param bar_length:
    :return:
    '''
    ## Data is a pandas dataframe
    timeseries720 = generate_timeseries(data, bar_length)
    df = pd.DataFrame(timeseries720)
    return df
