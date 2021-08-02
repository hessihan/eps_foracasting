# SARIMA module
# window forcast
# https://machinelearningmastery.com/simple-time-series-forecasting-models/

# import external modules
import sys
import warnings
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')

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
    def __init__(self, order, seasonal_order, y_prim_train, y_test, multivariate, x_prim_train, x_test, silent=True, store_models=False):
        self.models = []
        self.order = order
        self.seasonal_order = seasonal_order
        self.y_prim_train = y_prim_train
        self.multivariate = multivariate
        self.y_test = y_test
        self.x_prim_train = x_prim_train
        self.x_test = x_test
        self.silent = silent
        self.store_models = store_models
        
    def fit_rolling_window(self, window):
        """
        
        Iterate rolling window fitting and one period ahead forecasting for SARIMA model.
        
        Parameters
        ----------
        window: int
            Size of the rolling window.
            
        """
        y_pred = []
        pred_index = []
        for i in range(len(self.y_test)):
            # prepare temporary train data
            # expand training data
            y_temp_train = self.y_prim_train.append(self.y_test.iloc[:i])
            # drop the head of training data
            y_temp_train = y_temp_train[-window:]

            if self.multivariate:
                # prepare X for multivariate
                x_temp_train = self.x_prim_train.append(self.x_test.iloc[:i])
                x_temp_train = x_temp_train[-window:]
            else:
                x_temp_train=None

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
            if store_models:
                self.models.append(sarima)

            # describe if silent == False
            if not self.silent:
                print(sarima.summary())

                # residual ACF PACF
                resid = sarima.resid
                sm.graphics.tsa.plot_acf(resid)
                sm.graphics.tsa.plot_pacf(resid)

            # predict one period ahead
            pred_index.append(self.y_test.index[i])
            # always predict one step ahead (row number len(y_train)+1 = 41, (40 start from 0))
            if self.multivariate:
                y_pred.append(sarima.predict(window, exog=self.x_test.iloc[i]).values[0])
            else:
                y_pred.append(sarima.predict(window).values[0])
                
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