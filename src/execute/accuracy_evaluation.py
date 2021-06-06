# Evaluate predicted value (y_hats) for each methods.

# build forecasting accuracy indicators table
def accuracy_table(y_test, y_hats, indicators):
    """
    
    Return a forecasting accuracy indicators table for each y_hats of forecasting methods.
    
    Parameters
    ----------
    y_test : pd.Series
        The true data
    y_hats : dict
        The predicted data. {"name" : np.array}
    indicators : dict
        The forecasting accuracy functions. {"name" : function}
        Each function is called from src/utils/accuracy.py

    Returns
    -------
    pd.DataFrame
        Forecasting accuracy indicators table.
    
    """
    l = []
    # loop for each accuracy indicators
    for ind in indicators.keys():
        m = []
        # loop for each forecasting model
        for model in y_hats.keys():
            m.append(indicators[ind](y_test, y_hats[model]))
        l.append(m)
    
    a = pd.DataFrame(l, 
                     index=indicators.keys(),
                     columns=y_hats.keys()
                    )
    return a

if __name__ == "__main__":
    # import external packages
    import sys
    import numpy as np
    import pandas as pd
    
    # import internal modules
    sys.path.insert(1, '../')
    from utils.accuracy import *
    
    # read y_test cdv
    y_test = np.loadtxt("../../assets/y_hats/y_test.csv", delimiter=',', skiprows=1, usecols=1)
    
    # read y_hat csv
    y_hat_sarima_br = np.loadtxt("../../assets/y_hats/y_hat_sarima_br.csv", delimiter=',', skiprows=1, usecols=1)
    y_hat_sarima_g = np.loadtxt("../../assets/y_hats/y_hat_sarima_g.csv", delimiter=',', skiprows=1, usecols=1)
    y_hat_sarima_f = np.loadtxt("../../assets/y_hats/y_hat_sarima_f.csv", delimiter=',', skiprows=1, usecols=1)
    y_hat_mlp_mv = np.loadtxt("../../assets/y_hats/y_hat_mlp_mv.csv", delimiter=',', skiprows=1, usecols=1)
    
    # cumpute forecast accuracy indicators
    # accuracy function dict
    ind_name = ["MAE", "MAPE", "MSE", "RMSE", "RMSPE"]
    indicators = [MAE, MAPE, MSE, RMSE, RMSPE]
    indicators = dict(zip(ind_name, indicators))
    
    # y_hats dictionary
    method_name = ["SARIMA: BR", "SARIMA: G", "SARIMA: F", "MLP"]
    y_hats = [y_hat_sarima_br, y_hat_sarima_g, y_hat_sarima_f, y_hat_mlp_mv]
    y_hats = dict(zip(method_name, y_hats))
    
    a = accuracy_table(y_test, y_hats, indicators)
    a.to_csv("../../assets/accuracy_table_mv.csv")
    print(a)