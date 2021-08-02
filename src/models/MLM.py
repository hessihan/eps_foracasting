# Multivariate-Linear Model

# import external modules
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define MLM model class
class MLMModelList(object):
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    """
    
    The MLM model list for each prediction periods.
    
    MLM.1
    math::
        E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-4} + b_3 INV_{t-1} + b_4 AR_{t-1} + b_5 CAPX_{t-1} + b_6 GM_{t-1} + b_7 SA_{t-1} + b_8 ETR_{t-1} + b_9 LF_{t-1} + e_t
    
    MLM.2
    math::
        E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-4} + b_3 INV_{t-4} + b_4 AR_{t-4} + b_5 CAPX_{t-4} + b_6 GM_{t-4} + b_7 SA_{t-4} + b_8 ETR_{t-4} + b_9 LF_{t-4} + e_t

    Attributes
    ----------
    models : list
        list of `sm.tsa.statespace.SARIMAX()` objects to predict each periods.
    y_prim_train : pandas.Series
        primitive training data of y
    y_test : pandas.Series
        the remaining test data of y which is used to modify train data of y through window
    x_lag : int, 1 or 4
        number of lag for fundamental accounting variables.
    x_prim_train : pandas.Series
        primitive training data of x
    x_test : pandas.Series
        the remaining test data of x which is used to modify train data of x through window
    silent : bool (defualt True)
        False to allow outputting model summary and acf, pacf plot for model residual.
        
    """
    def __init__(self, y_prim_train, y_test, x_lag, x_prim_train, x_test, silent=True, store_models=False):
        self.models = []
        self.y_prim_train = y_prim_train
        self.y_test = y_test
        self.x_lag = x_lag
        self.x_prim_train = x_prim_train
        self.x_test = x_test
        self.silent = silent
        self.store_models = store_models
        
    def fit_rolling_window(self, window):
        """
        
        Iterate rolling window fitting and one period ahead forecasting for multivariate regression.
        
        Parameters
        ----------
        window: int
            Size of the rolling window.
            
        """
        # 一回 train_test_split してるのにもう一回統合してるのは非効率だけど
        # merge train and test for full period
        y_full = self.y_prim_train.append(self.y_test)
        x_full = self.x_prim_train.append(self.x_test)

        # lag X (and lag y as explanatpry variable)
        x_full = pd.concat([y_full.shift(1), y_full.shift(4), x_full.shift(self.x_lag)], axis=1)

        y_pred = []
        pred_index = []
        for i in range(len(self.y_test)):
            # get temporary train index
            temp_train_index = self.y_prim_train.index.append(self.y_test.index[:i])
            # drop the head of training data
            temp_train_index = temp_train_index[-window:]
                        
            # slice temporary train data
            y_temp_train = y_full.loc[temp_train_index]
            x_temp_train = x_full.loc[temp_train_index]
            
            # drop nan
            anynan_index = x_temp_train[x_temp_train.isna().any(axis=1)].index
            x_temp_train.drop(anynan_index, inplace=True)
            y_temp_train = y_temp_train.drop(anynan_index)
            
            # define and estimate Linear regression model from sklearn
            reg = LinearRegression(fit_intercept=True).fit(x_temp_train.values, y_temp_train.values.reshape(-1, 1))
            
            # append fitted model to list
            if self.store_models:
                self.models.append(reg)
                
            # get one step ahead test index
            one_step_ahead_test_index = self.y_test.index[i]
            pred_index.append(one_step_ahead_test_index)
            
            # predict one period ahead
            one_step_ahead_x_test = x_full.loc[one_step_ahead_test_index]
            
            # always predict one step ahead (row number len(y_train)+1 = 41, (40 start from 0))
            y_pred.append(reg.predict(one_step_ahead_x_test.values.reshape(1, -1)).item())
                
        pred_index = pd.MultiIndex.from_tuples(pred_index, names=self.y_test.index.names)
        y_hat = pd.Series(y_pred, index=pred_index, dtype=self.y_test.dtype)
        return y_hat
    
    def predict_window(self, ):
        """
        
        Return rolling window prediction of y from saved fitted model list.
        
        """
        y_hat = pd.Series([], dtype=test.dtype)
        # predict one period ahead
        for i in self.models:
            None
        return y_hat