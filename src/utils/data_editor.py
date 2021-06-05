import pandas as pd

def train_test_split(data, ratio=(4, 1)):
    """
    Time Series train test split (maintaining time order).
    
    Parameters
    ----------
    data : pd.DataFrame
        (periods, vars) shaped dataframe
    ratio : tuple
        (train, test) ratio. 
        
    Returns
    -------
    train_data, test_data : pd.DataFrame
        (train_periods, vars), (test_periods, vars) shaped two dataframes
    """
    # train test split (4/5) : (1/5), yearly bases for quaterly data
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

# create target and feature for NN
def target_feature():
    """
    Create target (model output: 1 dim) and feature (model input: multi dim) for NN.
    yとxは今のところearningかaccountingかをしめしてるので、
    targetとfeature(モデルのインプットとアウトプット)を指定しないといけない。
    テストの際もちょっとでーたがぐちゃるからそこら辺をやる。
    """

# tuple で inout sequence 作成 (training window 指定、サンプルサイズは len(input_data)-tw に減る)
def create_inout_sequence(input_data, tw):
    """
    """
    inout_seq = []
    for i in range(len(input_data)-tw):
        train_seq = input_data[i: i+tw]
        train_label = input_data[i+tw: i+tw+1]
        inout_seq.append((train_seq, train_label))
        
    return inout_seq

if __name__ == '__main__':
    # debug
    train_window = 5
    all_inout_seq = create_inout_sequence(all_normalized, train_window)