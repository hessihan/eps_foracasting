# SARIMA module
# window forcast
# https://machinelearningmastery.com/simple-time-series-forecasting-models/

# import external modules
import sys
import pandas as pd
import statsmodels.api as sm

# import internal modules

# Define SARIMA model class
# the object to store fitted models?????
class SARIMAModelList(object):
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    """
    
    The SARIMA model list for each prediction periods.

    Attributes
    ----------
    models : list
        list of `sm.tsa.statespace.SARIMAX()` objects to predict each periods.
    order : tuple
        (p, d, q)
        setting for `sm.tsa.statespace.SARIMAX()`
    seasonal_order : tuple
        (P, D, Q, S)
        setting for `sm.tsa.statespace.SARIMAX()`
    y_prim_train : pandas.Series
        primitive training data of y
    y_test : pandas.Series
        the remaining test data of y which is used to modify train data of y through window
    x_prim_train : pandas.Series
        primitive training data of x
    x_test : pandas.Series
        the remaining test data of x which is used to modify train data of x through window
    silent : bool (defualt True)
        False to allow outputting model summary and acf, pacf plot for model residual.
        
    """
    def __init__(self, order, seasonal_order, y_prim_train, y_test, x_prim_train, x_test, silent=True):
        self.models = []
        self.order = order
        self.seasonal_order = seasonal_order
        self.y_prim_train = y_prim_train
        self.y_test = y_test
        self.x_prim_train = x_prim_train
        self.x_test = x_test
        self.silent = silent
        
    def fit_rolling_window(self, window):
        """
        
        Iterate rolling window fitting and one period ahead forecasting for SARIMA model.
        
        Parameters
        ----------
        window: int
            Size of the moving window.
            
        """
        y_hat = pd.Series([], dtype=self.y_test.dtype)
        for i in range(len(self.y_test)):
            
            # prepare temporary train data
            # expand training data
            y_temp_train = self.y_prim_train.append(self.y_test.iloc[:i])
            x_temp_train = self.x_prim_train.append(self.x_test.iloc[:i])
            # drop the head of training data
            y_temp_train = y_temp_train[-window:]
            x_temp_train = x_temp_train[-window:]

            # define and estimate SARIMAX model from statsmodels
            sarima = sm.tsa.SARIMAX(
                endog=y_temp_train,
                exog=x_temp_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit()
            
            # append fitted model to list
            self.models.append(sarima)
            
            # describe if silent == False
            if not self.silent:
                print(sarima.summary())

                # residual ACF PACF
                resid = sarima.resid
                sm.graphics.tsa.plot_acf(resid)
                sm.graphics.tsa.plot_pacf(resid)
            
            # predict one period ahead
            next_period = y_temp_train.index.values[-1] + 1
            y_hat = y_hat.append(sarima.predict(start=next_period - i, # always train-size + 1 - 1 (start from zero) = window-size
                                                end=next_period - i,
                                                exog=self.x_test.loc[next_period]
                                               )
                                )
        return y_hat
    
    def predict_rolling_window(self, ):
        """
        
        Return rolling window prediction of y based on fitted model list.
        
        """
        y_hat = pd.Series([], dtype=test.dtype)
        # predict one period ahead
        for i in self.models:
            None
        return y_hat

# Expanding window forecast
def expanding_window(prim_train, test, order, seasonal_order, silent=True):
    """
    Iterate expanding window fitting and one period ahead forecasting for SARIMA model.
    
    Parameters
    ----------
    prim_train : pandas.Series
        primitive training data
    test : pandas.Series
        test data
    order : tuple
        (p, d, q)
    seasonal_order : tuple
        (P, D, Q, S)
    silent : bool
        False to allow outputting model summary and acf, pacf plot for model residual. (defualt True)
    """
    y_hat = pd.Series([], dtype=test.dtype)
    
    for i in range(len(test)):
        
        # expand training data
        train = prim_train.append(test.iloc[:i])
        
        # estimate model
        sarima = sm.tsa.SARIMAX(
            endog=train, 
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        
        # describe if silent == False
        if not silent:
            print(sarima.summary())
            
            # residual ACF PACF
            resid = sarima.resid
            sm.graphics.tsa.plot_acf(resid)
            sm.graphics.tsa.plot_pacf(resid)
        
        # predict one period ahead
        y_hat = y_hat.append(sarima.predict(train.index.values[-1] + 1))
    return y_hat

# Rolling window forecast
def rolling_window(window, prim_train, test, order, seasonal_order, silent=True):
    """
    Iterate rolling window fitting and one period ahead forecasting for SARIMA model.
    
    Parameters
    ----------
    window: int
        Size of the moving window.
    prim_train : pandas.Series
        primitive training data
    test : pandas.Series
        test data
    order : tuple
        (p, d, q)
    seasonal_order : tuple
        (P, D, Q, S)
    silent : bool
        False to allow outputting model summary and acf, pacf plot for model residual. (defualt True)
    """
    y_hat = pd.Series([], dtype=test.dtype)
    
    for i in range(len(test)):
        
        # expand training data
        train = prim_train.append(test.iloc[:i])
        # drop heads of training data
        train = train[-window:]
        
        # estimate model
        sarima = sm.tsa.SARIMAX(
            endog=train, 
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        
        # describe if silent == False
        if not silent:
            print(sarima.summary())
            
            # residual ACF PACF
            resid = sarima.resid
            sm.graphics.tsa.plot_acf(resid)
            sm.graphics.tsa.plot_pacf(resid)
        
        # predict one period ahead
        y_hat = y_hat.append(sarima.predict(train.index.values[-1] + 1))
        
    return y_hat

# Debug
if __name__ == "__main__":

    # from moving_window import 

    # read processed dataset generated from data_preprocessing.py
    ts = pd.read_csv("../../data/processed/dataset.csv", index_col=0)
    
    # provide y and x (lag 1 y, lag 1 x)
    # !!!!!! lag 4 for quarterly ? or statsmodel have done well in sm.tsa.SARIMAX?
    y = ts["１株当たり利益"]

    # train test split (4 : 1)
    y_train, y_test = train_test_split(data=y, ratio=(4, 1))
    
    # Define unfitted SARIMA models
    sarima_br = SARIMA(y_train, None, (1, 0, 0), (0, 1, 1, 4))
    
    # expanding window
    y_hat_expanding = expanding_window(y_train, y_test, (1, 0, 0), (0, 1, 1, 4), silent=True)
    
    # Rolling window
    y_hat_rolling = rolling_window(40, y_train, y_test, (1, 0, 0), (0, 1, 1, 4), silent=True)