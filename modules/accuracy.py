# Forecast Accuracy Indicator

# https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d
# https://qiita.com/23tk/items/2dbb930242eddcbb75fb

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# MAE
def MAE(true, pred):
    "Mean Absolute Error"
    return mean_absolute_error(true, pred)

# MAPE
def MAPE(true, pred):
    "Mean Absolute Percentage Error"
    return np.mean(np.abs((pred - true) / true))

# MSE
def MSE(true, pred):
    "Mean Squared Error"
    return mean_squared_error(true, pred)

# RMSE
def RMSE(true, pred):
    "Root Mean Squared Error"
    return np.sqrt(mean_squared_error(true, pred))
    
# RMSPE
def RMSPE(true, pred):
    "Root Mean Squared Percentage Error"
    return np.sqrt(np.mean(((pred - true) / true)**2))