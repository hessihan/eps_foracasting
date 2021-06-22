import numpy as np
import pandas as pd
import torch

def train_test_split(data, test_size=None, ratio=(4, 1)):
    """
    Time Series train test split (maintaining time order).
    
    Parameters
    ----------
    data : pd.DataFrame
        (periods, vars) shaped dataframe
    test_size : int
        Absolute test data size.
    ratio : tuple
        (train, test) ratio. 
        
    Returns
    -------
    train_data, test_data : pd.DataFrame
        (train_periods, vars), (test_periods, vars) shaped two dataframes
    """
    if test_size != None:
    # split test with absolute size
        pass
    else:
    # train test split with ratio (4/5) : (1/5), yearly bases for quaterly data
        test_size = ratio[1] * (len(data) // sum(ratio)) + (len(data) // sum(ratio)) % ratio[0]
        train_size = len(data) - test_size
    
    return data.iloc[:-test_size], data.iloc[-test_size:]

def lag(data, lag, drop_nan=False, reset_index=False):
    """
    Create lagged data for NN.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        (periods, vars) shaped dataframe
    lag : int
        number of lags
    drop_nan : bool, defalt False
        drop the row including NaN
    reset_index : bool, defalt False
        Reset the index after dropping rows. It is not recommended to reset original index because you will never find the original row for the lagged data.

    Returns
    -------
    data_lag : pd.DataFrame
        (data.shape[0] - lag) x (data.shape[1] * lag) shaped pd.DataFrame
    """
    data_lag = pd.DataFrame() # ここは横にconcatしないと、DataFrameに対応できない。
    for i in range(lag):
        data = pd.DataFrame(data)
        shift = data.shift(i+1)
        shift.columns = [colname + "_lag" + str(i+1) for colname in shift.columns]
        data_lag = pd.concat([data_lag, shift], axis=1)

    if drop_nan :
        data_lag = data_lag.drop([i for i in range(lag)], axis=0)

    if reset_index :
        data_lag = data_lag.reset_index(drop=True)

    return data_lag

# Unlike MLP, LSTM needs to prepare lagged inputs with seq_len matrix.
# inherit from the torch.utils.data.Dataset class
class TimeseriesDataset(torch.utils.data.Dataset):   
    """
    Torch based time-series dataset object class.
    This object could be the input for torch.utils.data.DataLoader() when running LSTM. 
    For MLP, you might have to squeeze the output matrix of DataLoader to vector.

    Parameters
    ----------
    feature_lag1 : torch.tensor
        explanatory variables matrix with only 1 lag.
    feature_lag1 : torch.tensor
        explained variable vector with no lag.
    seq_len : int
        There're so many names for this.
            * rolling window size
            * sliding window size
            * training window size
            * sequence length
        Anyway, this is the number of window lags of inputs for one step ahead prediction.
    
    Reference:
    https://stackoverflow.com/questions/57893415/pytorch-dataloader-for-time-series-task
    """
    def __init__(self, feature_lag1, target, seq_len=None):
        self.feature_lag1 = feature_lag1
        self.target = target
        self.seq_len = seq_len

    def __len__(self):
        return self.feature_lag1.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.feature_lag1[index:index+self.seq_len], self.target[index+self.seq_len-1])

# create target and features for NN
def target_feature_beta():
    """
    Create target (model output: 1 dim) and feature (model input: multi dim) for NN.
    yとxは今のところearningかaccountingかをしめしてるので、
    targetとfeature(モデルのインプットとアウトプット)を指定しないといけない。
    テストの際もちょっとデータがぐちゃるからそこら辺をやる。
    """

# tuple で inout sequence 作成 (training window 指定、サンプルサイズは len(input_data)-tw に減る)
def window_sequence_beta(table_data, window_size):
    """
    Input original normal table time-series data and return a list of sliding windows 
    in tuple shape (train_window, train_point), which is rolling window format.
    
    https://stackoverflow.com/questions/53791838/elegant-way-to-generate-indexable-sliding-window-timeseries
    https://stackoverflow.com/questions/57893415/pytorch-dataloader-for-time-series-task
    https://stackoverflow.com/questions/47482009/pandas-rolling-window-to-return-an-array
    
    Parameters
    ----------
    table_data : pd.DataFrame or pd.Series
        (periods, vars) shaped normal time-series dataframe
    window_size : int
        size of rolling (sliding, training) window for a single sequence

    Returns
    -------
    seq_list : list of tuples
        a tuple is containing a rolling_window and label set. (train_window, train_point)
        
    """
    seq_list = []
    for i in range(len(input_data)-window_size):
        train_window = input_data[i: i+window_size]
        train_point = input_data[i+window_size: i+window_size+1]
        inout_seq.append((train_window, train_point))
        
    return seq_list

if __name__ == '__main__':
    # debug
    train_window = 5
    all_inout_seq = create_inout_sequence(all_normalized, train_window)