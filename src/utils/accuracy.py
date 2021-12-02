# Forecast Accuracy Indicator

# https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d
# https://qiita.com/23tk/items/2dbb930242eddcbb75fb

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error

# vector to scalar metrics

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

# sample level (vec2vec, scalar2scalar) error metrics

# exact error
def Error(true, pred):
    """
    Exact error
    
    Return
    ------
    true - pred
    """
    return true - pred

# absolute error
def AbsoluteError(true, pred):
    """
    Exact error
    
    Return
    ------
    abs(true - pred)
    """
    return abs(true - pred)

# percentage error
def PercentageError(true, pred):
    """
    Percentage error
    
    Return
    ------
    (true - pred) / true
    """
    return (true - pred) / true

# absolute percentage error
def AbsolutePercentageError(true, pred):
    """
    Absolute percentage error
    
    Return
    ------
    abs((true - pred) / true)
    """
    return abs((true - pred) / true)

# absolute percentage error with upper bound
def AbsolutePercentageErrorUB(true, pred):
    """
    Absolute percentage error with upper bound
    
    Return
    ------
    abs((true - pred) / true) or 1
    """
    ape_ub = AbsolutePercentageError(true, pred)
    ape_ub[ape_ub > 1] = 1
    return ape_ub

def LargeErrorRate(true, pred, percent=True):
    """
    Percentage of large error forecast
    """
    ape_ub = AbsolutePercentageError(true, pred)
    lfe = np.mean(ape_ub > 1)
    if percent:
        lfe *= 100        
    return lfe

def MAPEUB(true, pred):
    """
    Mean Absolute Percentage Error with upper bound
    """
    return np.mean(AbsolutePercentageErrorUB(true, pred))

def MSPEUB(true, pred):
    """
    Mean Squared Percentage Error with upper bound
    """
    return np.mean(AbsolutePercentageErrorUB(true, pred)**2)

def get_non_late_firm():
    tidy_df = pd.read_csv("/mnt/d/0ngoing/thesis/repo/data/processed/tidy_df.csv", index_col=[0, 1, 2])
    all_firm = tidy_df.index.get_level_values(0).unique()
    # Check release date
    tidy_tse1 = pd.read_csv("/mnt/d/0ngoing/thesis/repo/data/processed/tidy_tse1.csv", index_col=[0, 1, 2])
    release_date = tidy_tse1["決算発表日"]
    release_date = pd.to_datetime(release_date)

    release_date_test_period = release_date.loc[pd.IndexSlice[all_firm, [2017, 2018, 2019, 2020], :]]
    release_date_test_period = release_date_test_period.drop(release_date_test_period.loc[pd.IndexSlice[:, [2017], ["Q1", "Q2", "Q3"]]].index)
    release_date_test_period = release_date_test_period.drop(release_date_test_period.loc[pd.IndexSlice[:, [2020], ["Q4"]]].index)

    # 各対象四半期の最終月(6月, 9月, 12月, 3月)に発表だと遅い。IBESは(6月1日, 9月1日, 12月1日, 3月1日)時点の予測で統一。
    release_date_test_period.groupby(release_date_test_period.dt.month).count()

    release_date_test_period[release_date_test_period.dt.month == 6]
    release_date_test_period[release_date_test_period.dt.month == 9] # 理研ビタミン 2019  Q4  2020-09-30 <-- 1四半期遅れ
    release_date_test_period[release_date_test_period.dt.month == 12]
    release_date_test_period[release_date_test_period.dt.month == 3]

    # report release date for each quarter
    def get_late_firm(y, q):
        # print("/// month count ///")
        # print(release_date_test_period.loc[pd.IndexSlice[:, y, q]].groupby(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month).count())
        if q == "Q1":
            m1, m2 = 7, 8
        elif q == "Q2":
            m1, m2 = 10, 11
        elif q == "Q3":
            m1, m2 = 1, 2
        elif q == "Q4":
            m1, m2 = 4, 5
        # print("/// late firm ///")
        # print(release_date_test_period.loc[pd.IndexSlice[:, y, q]][~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m1) & ~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m2)])
        return release_date_test_period.loc[pd.IndexSlice[:, y, q]][~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m1) & ~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m2)].index

    late_firm = []
    for i in [
        [2017, "Q4"], [2018, "Q1"], [2018, "Q2"], [2018, "Q3"], 
        [2018, "Q4"], [2019, "Q1"], [2019, "Q2"], [2019, "Q3"], 
        [2019, "Q4"], [2020, "Q1"], [2020, "Q2"], [2020, "Q3"]
        ]:
        late_firm += list(get_late_firm(i[0], i[1]))
    late_firm = list(set(late_firm))
    non_late_firm = [x for x in all_firm if (x not in late_firm)]
    return non_late_firm
