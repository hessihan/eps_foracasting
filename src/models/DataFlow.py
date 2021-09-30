# Automate firm loop, rolling sample, train-valid-test split

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(1, '../')
from utils.data_editor import train_test_split, lag

def lagged_data(n):
    lag_n = df.groupby(level=0).shift(n)
    lag_n.columns = [i + "_-" + str(n) for i in col]
    return lag_n

def dataflow_for(func, df, firm_list, rolling_sample_size): #, train_size, valid_size, test_size):
    """
    --- For Loop Ver. ---
    Automate firm loop, rolling sample, train-valid-test split.
    Return output of func (the input) for each data flow.
    """
    l = []
    # firm loop
    for i in firm_list:
        df_i = df.loc[i]
        df_i = pd.concat([df_i, lag(df_i, 4)])
        df_i.dropna(inplace=True)
        l2 = []
        # rolling sample loop
        for j in range(rolling_sample_size):
            df_ij = df_i.iloc[j : j + (len(df_i) - rolling_sample)]
            l2.append(func(df_ij))
        l.append(l2)
    return l

# loop multiindex
def dataflow_single_loop():
    return None

def get_firm_rolling_sample(i, j):
    """
    i: firm
    j: rolling_sample
    func: core function to apply df_ij
    """
    df_i = df.loc[i]
    df_i = pd.concat([df_i, lag(df_i, 4)])
    df_i.dropna(inplace=True)
    df_ij = df_i.iloc[j : j + (len(df_i) - rolling_sample)]
    return nothing(df_ij)
            
def dataflow_map(func, df, firm_list, rolling_sample_size): #, train_size, valid_size, test_size):
    """
    --- Map Ver. ---
    Automate firm loop, rolling sample, train-valid-test split.
    Return output of func (the input) for each data flow.
    """
    
    # map firm
    l = map(functools.partial(get_firm_rolling_sample, func=func), firm_list, range(rolling_sample_size))

        
def lagged_data(df, col, n):
    lag_n = df[col].groupby(level=0).shift(n)
    lag_n.columns = [i + "_lag" + str(n) for i in col]
    return lag_n

# Debug
if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    col = ['EPS', 'INV', 'AR', 'CAPX','GM', 'SA', 'ETR', 'LF']
    df = pd.concat([
        df,
        lagged_data(df, col, 1),
        lagged_data(df, col, 2),
        lagged_data(df, col, 3),
        lagged_data(df, col, 4),
    ], axis=1)
    df.dropna(inplace=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    col = ['EPS', 'INV', 'AR', 'CAPX','GM', 'SA', 'ETR', 'LF']
    df = df[col]
    firm_list = df.index.get_level_values(0).unique()
    rolling_sample_size = 12
    
    def get_shape(df_ij):
        return df_ij.shape
    
    def nothing(df_ij):
        return df_ij
    
    t1 = time.time()
    l = dataflow_for(get_shape, df, firm_list, 12)
    t2 = time.time()
    print(t2 - t1)
    
    t1 = time.time()
    l = list(map(get_firm_df, firm_list[:10]))
    t2 = time.time()
    print(t2 - t1)
    
    list(map(get_firm_rolling_sample, firm_list[:2], range(rolling_sample_size)))
