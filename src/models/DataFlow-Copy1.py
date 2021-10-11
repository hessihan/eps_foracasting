# Automate firm loop, rolling sample, train-valid-test split

import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(1, '../')
from utils.data_editor import train_test_split, lag
from utils.accuracy import *

import warnings
warnings.simplefilter('ignore')

def lagged_data(df, col, n):
    lag_n = df[col].groupby(level=0).shift(n)
    lag_n.columns = [i + "_lag" + str(n) for i in col]
    return lag_n

def dataflow_for(func, df, firm_list, test_periods): #, train_size, valid_size, test_size):
    """
    --- For Loop Ver. ---
    Automate firm loop, rolling sample, train-valid-test split.
    Return output table (firm_list * test_periods) of func (the input) for each data flow.
    One step ahead forecast for each test_periods
    """
    t1 = time.time()
    # empty dataframe
    output = pd.DataFrame(index=df.loc[pd.IndexSlice[:, 2018:, :], :].index, columns=["y_hat"])
    # firm loop 
    for i in tqdm(firm_list):
        df_i = df.loc[i]
        # rolling sample loop
        for num, j in enumerate(test_periods):
            df_ij = df_i.iloc[num: num + (len(df_i) - len(test_periods) + 1)]
            # train-valid-test split
            y_test = df_ij.loc[j][["EPS"]]                    # (1,)     1D Series
            x_test = pd.DataFrame(df_ij.loc[j].drop("EPS")).T # (1, 39)  2D DataFrame
            y_train = df_ij.drop(j)["EPS"]                    # (36,)    1D Series
            x_train = df_ij.drop(j).drop("EPS", axis=1)       # (36, 39) 2D DataFrame
            output.loc[i, j[0], j[1]] = func(x_train, y_train, x_test)
    t2 = time.time()
    print("Elapsed Time: ", t2-t1)
    return output

def dataflow_for_(func, df, firm_list, test_periods: int, val_size: int):
    """
    --- For Loop Ver. ---
    Automate firm loop, rolling sample, train-valid-test split.
    Return output table (firm_list * test_periods) of func (the input) for each data flow.
    One step ahead forecast for each test_periods.
    split train_val to train and val
    """
    t1 = time.time()
    # empty dataframe
    output = pd.DataFrame(index=df.loc[pd.IndexSlice[:, 2018:, :], :].index, columns=["y_hat"])
    # firm loop 
    for i in tqdm(firm_list):
        df_i = df.loc[i]
        # rolling sample loop
        for num, j in enumerate(test_periods):
            df_ij = df_i.iloc[num: num + (len(df_i) - len(test_periods) + 1)]
            # train-valid-test split
            y_test = df_ij.loc[j][["EPS"]]                    # (1,)     1D Series
            x_test = pd.DataFrame(df_ij.loc[j].drop("EPS")).T # (1, 39(num_x))  2D DataFrame
            # No validation
            if val_size <= 0:
                y_train = df_ij.drop(j)["EPS"]                    # (36,)    1D Series
                x_train = df_ij.drop(j).drop("EPS", axis=1)       # (36, 39) 2D DataFrame
                output.loc[i, j[0], j[1]] = func(x_train, y_train, x_test)
            # with validation
            else:
                y_train_val = df_ij.drop(j)["EPS"]                      # (36,)    1D Series
                x_train_val = df_ij.drop(j).drop("EPS", axis=1)         # (36, 39) 2D DataFrame
                y_val = y_train_val.iloc[-val_size:]                    # (val_size,)    1D Series
                x_val = x_train_val.iloc[-val_size:]                    # (val_size, 39) 2D DataFrame
                y_train = y_train_val.iloc[:-val_size]                  # (36-val_size,)    1D Series
                x_train = x_train_val.iloc[:-val_size]                  # (36-val_size, 39) 2D DataFrame
                output.loc[i, j[0], j[1]] = func(x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test)
    t2 = time.time()
    print("Elapsed Time: ", t2-t1)
    return output
    
########## Mapping (Ongoing) #############################################################

# loop multiindex
def dataflow_for_single(func, df, firm_list, test_periods): #, train_size, valid_size, test_size):
    """
    --- Single For Loop Ver. ---
    Automate firm loop, rolling sample, train-valid-test split.
    Return output table (firm_list * test_periods) of func (the input) for each data flow.
    """
    t1 = time.time()
    # empty dataframe
    output = pd.DataFrame(index=df.loc[pd.IndexSlice[:, 2018:, :], :].index, columns=["y_hat"])
    df_reindex = df.reset_index()
    # firm-rolling sample loop
    for k in tqdm(output.index):
        test_index = df_reindex[(df_reindex["企業名"] == k[0]) & (df_reindex["会計年度"] == k[1]) & (df_reindex["四半期"] == k[2])].index
        df_ij = df_reindex.iloc[test_index[0] - ( len(df_reindex[df_reindex["企業名"]==k[0]]) - len(test_periods) ): test_index[0]+1]
        df_ij.set_index(["企業名", "会計年度", "四半期"], inplace=True)
        # train-valid-test split
        y_test = df_ij.loc[k][["EPS"]]                  # (1,)     1D Series
        x_test = pd.DataFrame(df_ij.loc[k].drop("EPS")).T # (1, 39)  2D DataFrame
        y_train = df_ij.drop(k)["EPS"]                    # (36,)    1D Series
        x_train = df_ij.drop(k).drop("EPS", axis=1)       # (36, 39) 2D DataFrame
        output.loc[k] = func(x_train, y_train, x_test)
    t2 = time.time()
    print("Elapsed Time: ", t2-t1)
    return output

def get_firm_rolling_sample(i, j):
    """
    i: firm
    j: rolling_sample
    func: core function to apply df_ij
    """
    df_i = df.loc[i]
    df_i = pd.concat([df_i, lag(df_i, 4)])
    df_i.dropna(inplace=True)
    df_ij = df_i.iloc[j : j + (len(df_i) - rolling_sample)]
    return nothing(df_ij)
            
def dataflow_map(func, df, firm_list, rolling_sample_size): #, train_size, valid_size, test_size):
    """
    --- Map Ver. ---
    Automate firm loop, rolling sample, train-valid-test split.
    Return output of func (the input) for each data flow.
    """
    
    # map firm
    l = map(functools.partial(get_firm_rolling_sample, func=func), firm_list, range(rolling_sample_size))
#######################################################################

# Debug
if __name__ == "__main__":
    
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    import lightgbm as lgb
    import optuna
    
    def moc_func(x_train, y_train, x_test):
        return y_train.mean() 

    def moc_func_val(x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test):
        return (x_train.shape[0] + x_val.shape[0] == x_train_val.shape[0]) & (y_train.shape[0] + y_val.shape[0] == y_train_val.shape[0])

    def RAF(x_train, y_train, x_test):  
        model = RandomForestRegressor(n_estimators=100, max_depth=None, max_features="auto", random_state=0)
        model.fit(x_train, y_train)
        # in-sample fit --> need outer function
    #         model.predict(x_train)
        # out-sample fit
        y_hat = model.predict(x_test)[0]
        return y_hat

    def L2(x_train, y_train, x_test):
        model = Ridge(alpha=10)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)[0]
        return y_hat

    def L1_spoiled(x_train, y_train, x_test):
        model = Lasso(alpha=10)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)[0]
        return y_hat

    def EN(x_train, y_train, x_test):
        model = ElasticNet(alpha=10, l1_ratio=0.5)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)[0]
        return y_hat

    def SVM(x_train, y_train, x_test):
        model = SVR()
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)[0]
        return y_hat

    def LGB(x_train, y_train, x_test):
        model = lgb.LGBMRegressor()
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)[0]
        return y_hat

    def L1(x_train, y_train, x_test, alpha):
        model = Lasso(alpha=alpha)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
        y_fitted = pd.DataFrame(model.predict(x_train), index=x_train.index)
        return y_hat, y_fitted
    
    def objective(trial):
        # hyper paramater tuning space
        alpha = trial.suggest_float("alpha", 0.0, 2.0)

        # モデルの学習と評価
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)
        model = sklearn.linear_model.Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        error = sklearn.metrics.mean_squared_error(y_val, y_pred)

        # 平均二乗誤差を目的関数からの出力とする
        return error    
        
    def machine_learning(x_train, y_train, x_val, y_val, x_train_val, y_train_val, x_test):
        
        model = Lasso(alpha=1.0)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)[0]
        # Extra arguments.
        min_x = -100
        max_x = 100

        # Execute an optimization by using the above objective function wrapped by `lambda`.
        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, min_x, max_x), n_trials=100)
        return y_hat
    
    
    
    # prepare lagged data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    col = ['EPS', 'INV', 'AR', 'CAPX','GM', 'SA', 'ETR', 'LF']
    df = df[col]
    df = pd.concat([
        df,
        lagged_data(df, col, 1),
        lagged_data(df, col, 2),
        lagged_data(df, col, 3),
        lagged_data(df, col, 4),
    ], axis=1)
    df.dropna(inplace=True)
    
#     df = df[col]
    firm_list = df.index.get_level_values(0).unique()
    rolling_sample_size = 12
    test_periods = [(i, j) for i in [2018, 2019, 2020] for j in ["Q1", "Q2", "Q3", "Q4"]]

    # apply model for each firm-rolling sample
    y_hat_moc = dataflow_for(moc_func, df, firm_list[:10], test_periods, 0)
    y_hat_moc_val = dataflow_for(moc_func_val, df, firm_list[:10], test_periods, 4)
#     y_hat_moc_single = dataflow_for_single(moc_func, df, firm_list, test_periods) # おそい(set_indexしてるから?)
    
    y_hat_mraf = dataflow_for(RAF, df, firm_list, test_periods)
    y_hat_mraf.columns=["y_hat_mraf"]
#     y_hat_mraf.to_csv("../../assets/y_hats/multivariate/y_hat_mraf_default.csv")
    
    y_hat_ml2 = dataflow_for(L2, df, firm_list, test_periods)
    y_hat_ml2.columns=["y_hat_ml2"]
#     y_hat_ml2.to_csv("../../assets/y_hats/multivariate/y_hat_ml2_default.csv")
    
    y_hat_ml1 = dataflow_for(L1, df, firm_list, test_periods)
    y_hat_ml1.columns=["y_hat_ml1"]
#     y_hat_ml1.to_csv("../../assets/y_hats/multivariate/y_hat_ml1_default.csv")
    
    y_hat_men = dataflow_for(EN, df, firm_list, test_periods)
    y_hat_men.columns=["y_hat_men"]
#     y_hat_men.to_csv("../../assets/y_hats/multivariate/y_hat_men_default.csv")
    
    y_hat_mlgb = dataflow_for(LGB, df, firm_list, test_periods)
    y_hat_mlgb.columns=["y_hat_mlgb"]
#     y_hat_mlgb.to_csv("../../assets/y_hats/multivariate/y_hat_mlgb_default.csv")
    
    y_hat_msvm = dataflow_for(SVM, df, firm_list, test_periods)
    y_hat_msvm.columns=["y_hat_msvm"]
#     y_hat_msvm.to_csv("../../assets/y_hats/multivariate/y_hat_msvm_default.csv")
    
    y_test = pd.read_csv("../../assets/y_hats/univariate/y_test.csv", index_col=[0, 1, 2])
    
    MAPEUB(y_test.values, y_hat_mlgb.values)
    LargeErrorRate(y_test.values, y_hat_mlgb.values)

    MAPEUB(y_test.values, y_hat_msvm.values)
    LargeErrorRate(y_test.values, y_hat_msvm.values)
    
    # Canhe hyper params
    y_hat_ml2_a10 = dataflow_for(L2, df, firm_list, test_periods)
    y_hat_ml2_a10.columns=["y_hat_ml2_a10"]
#     y_hat_ml2_a10.to_csv("../../assets/y_hats/multivariate/y_hat_ml2_a10.csv")
    MAPEUB(y_test.values, y_hat_ml2_a10.values)
    LargeErrorRate(y_test.values, y_hat_ml2_a10.values)

    y_hat_ml1_a10 = dataflow_for(L1_spoiled, df, firm_list, test_periods)
    y_hat_ml1_a10.columns=["y_hat_ml1_a10"]
#     y_hat_ml1_a10.to_csv("../../assets/y_hats/multivariate/y_hat_ml1_a10.csv")
    MAPEUB(y_test.values, y_hat_ml1_a10.values)
    LargeErrorRate(y_test.values, y_hat_ml1_a10.values)

    y_hat_men_a10 = dataflow_for(EN, df, firm_list, test_periods)
    y_hat_men_a10.columns=["y_hat_men_a10"]
#     y_hat_men_a10.to_csv("../../assets/y_hats/multivariate/y_hat_men_a10.csv")
    MAPEUB(y_test.values, y_hat_men_a10.values)
    LargeErrorRate(y_test.values, y_hat_men_a10.values)
