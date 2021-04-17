# SARIMA module
# window forcast
# https://machinelearningmastery.com/simple-time-series-forecasting-models/

import pandas as pd
import statsmodels.api as sm

# Expanding window forecast
def expanding_window(prim_train, test, order, seasonal_order, silent=True):
    """
    Iterate expanding window forecasting using `sm.tsa.SARIMAX()`.
    
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
    Iterate rolling window forecasting using `sm.tsa.SARIMAX()`.
    
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

# Debugging
if __name__ == "__main__":
    # read cleaned data generated from data_preprocessing.py
    ts = pd.read_csv("data/cleaned/sample_ts.csv", index_col=0)

    # y, x (lag 1 y) 
    # !!!!!! lag 4 for quarterly ? or statsmodel have done well in sm.tsa.SARIMAX?
    y = ts["１株当たり利益［３ヵ月］"]

    # train (40) test (11) split
    y_train = y[:40]
    y_test = y[40:]
    
    # expanding window
    y_hat_expanding = expanding_window(y_train, y_test, (1, 0, 0), (0, 1, 1, 4), silent=True)
    
    # Rolling window
    y_hat_rolling = rolling_window(40, y_train, y_test, (1, 0, 0), (0, 1, 1, 4), silent=True)