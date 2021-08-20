# Forecast Accuracy Indicator

# https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d
# https://qiita.com/23tk/items/2dbb930242eddcbb75fb

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error

# Max error
def Max_error(true, pred):
    "Maximum absolute residual error"
    return max_error(true, pred)

# Max percentage error
def Max_percentage_error(true, pred):
    "Maximum absolute percentage residual error"
    return np.max(np.abs((true - pred) / true))

# MAE
def MAE(true, pred):
    "Mean Absolute Error"
    return mean_absolute_error(true, pred)

# MAPE
def MAPE(true, pred):
    """
    Mean Absolute Percentage Error
    
    Error is calculated with rate.
    """
    return np.mean(np.abs((pred - true) / true))

# MSE
def MSE(true, pred, rate=True):
    """
    Mean Squared Error
    
    Parameters
    ----------
    rate : bool, default=True
        Calculate MSE with rate of error if True.
        
        math'''
            \frac {1} {N} \sum^{N}_{i=1}( \frac {(Y_t - \hat{y}_t)} {Y_t} )^2
        '''
        
        if False, calculate ordinary MSE.
        
        math'''
            \frac {1} {N} \sum^{N}_{i=1}(Y_t - \hat{y}_t)^2
        '''
    """
    if rate:
        mse = np.mean(((true - pred) / true)**2)
    else:
        mse = mean_squared_error(true, pred)
    return mse

# RMSE
def RMSE(true, pred):
    """
    Root Mean Squared Error
    
    Parameters
    ----------
    rate : bool
        RMSE may also need to be calculated with rate of error
    """
    return np.sqrt(mean_squared_error(true, pred))
    
# RMSPE
def RMSPE(true, pred):
    "Root Mean Squared Percentage Error"
    return np.sqrt(np.mean(((pred - true) / true)**2))