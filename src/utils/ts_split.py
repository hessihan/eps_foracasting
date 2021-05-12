# Time Series train test split (maintaining time order)

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