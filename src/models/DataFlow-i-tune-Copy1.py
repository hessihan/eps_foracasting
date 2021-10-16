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

def dataslice(df, firm_list, test_periods, val_size, i, j):
    """
    get specified splitted datasets
    
    i: firm
    j: single_test_period (rolling_sample)
    """
    df_i = df.loc[i]
    df_ij = df_i.loc[: j].iloc[-(len(df_i)-len(test_periods)+1):]
    # train-valid-test split
    x_train_val = df_ij.drop(j).drop("EPS", axis=1)
    y_train_val = df_ij.drop(j)["EPS"]
    splitted_data = {
        "x_test": pd.DataFrame(df_ij.loc[j].drop("EPS")).T,       # (1, 39(num_x))  2D DataFrame
        "y_test": df_ij.loc[j][["EPS"]],                          # (1,)     1D Series
        "x_train_val": x_train_val,                               # (36, 39) 2D DataFrame
        "y_train_val": y_train_val,                               # (36,)    1D Series
        "x_val": x_train_val.iloc[-val_size:],                    # (val_size, 39) 2D DataFrame
        "y_val": y_train_val.iloc[-val_size:],                    # (val_size,)    1D Series
        "x_train": x_train_val.iloc[:-val_size],                  # (36-val_size, 39) 2D DataFrame
        "y_train": y_train_val.iloc[:-val_size],                  # (36-val_size,)    1D Series
    }
    return splitted_data

def split_data_i(firm):        
    splitted_data_i = {
        "x_test": [], "y_test": [],
        "x_train_val": [], "y_train_val": [], 
        "x_val": [], "y_val": [], 
        "x_train": [], "y_train": []
    }
    for j in test_periods:
        df_ij = dataslice(df, firm_list, test_periods, val_size=1, i=firm, j=j)
        splitted_data_i["x_test"].append(df_ij["x_test"])
        splitted_data_i["y_test"].append(df_ij["y_test"])
        splitted_data_i["x_train_val"].append(df_ij["x_train_val"])
        splitted_data_i["y_train_val"].append(df_ij["y_train_val"])
        splitted_data_i["x_val"].append(df_ij["x_val"])
        splitted_data_i["y_val"].append(df_ij["y_val"])
        splitted_data_i["x_train"].append(df_ij["x_train"])
        splitted_data_i["y_train"].append(df_ij["y_train"])
    return splitted_data_i

def L1_i_tune(splitted_data_i, tune_space):
    from sklearn.linear_model import Lasso
    def L1(x_train, y_train, x_test, alpha):
        model = Lasso(alpha=alpha, random_state=0)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
        y_fitted = pd.Series(model.predict(x_train), index=x_train.index)
        return y_hat, y_fitted

    def valfit_map(alpha):
        y_val_hat = np.arange(len(test_periods))
        y_val_hat = list(map(lambda period: L1(splitted_data_i["x_train"][period], splitted_data_i["y_train"][period], splitted_data_i["x_val"][period], alpha=alpha)[0], y_val_hat))
        y_val_hat = pd.concat(y_val_hat)
        return y_val_hat

    def validation_map(tune_space):
        y_val = pd.concat(splitted_data_i["y_val"])
        tune_space = tune_space
        val_mspe = np.array(list(map(lambda a: MSE(y_val, valfit_map(a)), tune_space)))
        best_hyparam = tune_space[val_mspe.argmin()]
#         print("best hyparam:", best_hyparam)
#         print("min val MSPE:", val_mspe.min())
        return best_hyparam

    def trainvalfit_map(alpha):
        y_hat = np.arange(len(test_periods))
        y_hat = list(map(lambda period: L1(splitted_data_i["x_train_val"][period], splitted_data_i["y_train_val"][period], splitted_data_i["x_test"][period], alpha=alpha)[0], y_hat))
        y_hat = pd.concat(y_hat)
        return y_hat
    
#     import matplotlib.pyplot as plt
#     plt.plot(np.log10(tune_space), val_mspe)
#     plt.plot(tune_space, val_mspe)
    best_alpha = validation_map(tune_space)
    y_hat = trainvalfit_map(best_alpha)
    return y_hat

def i_tune(splitted_data_i, method, tune_space):
    """
    splitted_data_i: dict
        use return of split_data_i()
    method: func
        model function method(x_train, y_train, x_test, hyparams) which returns y_hat 
    tune space: array or list
        1D array (multiple tuning parameters should be squased, flattened)
    """

    def prim_train(hyparam):
        y_val_hat = np.arange(len(test_periods))
        y_val_hat = list(map(lambda period: method(splitted_data_i["x_train"][period], splitted_data_i["y_train"][period], splitted_data_i["x_val"][period], hyparam), y_val_hat))
        y_val_hat = pd.concat(y_val_hat)
        return y_val_hat

    def search_best_hyparam(tune_space):
        y_val = pd.concat(splitted_data_i["y_val"])
        tune_space = tune_space
        val_mspe = np.array(list(map(lambda hyparam: MSE(y_val, prim_train(hyparam)), tune_space)))
        best_hyparam = tune_space[val_mspe.argmin()]
#         print("best hyparam:", best_hyparam)
#         print("min val MSPE:", val_mspe.min())
        return best_hyparam

    def final_train(best_hyparam):
        y_hat = np.arange(len(test_periods))
        y_hat = list(map(lambda period: method(splitted_data_i["x_train_val"][period], splitted_data_i["y_train_val"][period], splitted_data_i["x_test"][period], best_hyparam), y_hat))
        y_hat = pd.concat(y_hat)
        return y_hat
    
#     import matplotlib.pyplot as plt
#     plt.plot(np.log10(tune_space), val_mspe)
#     plt.plot(tune_space, val_mspe)
    best_hyparam = search_best_hyparam(tune_space)
    y_hat = final_train(best_alpha)
    return y_hat
#######################################################################

# Debug
if __name__ == "__main__":
    
    from sklearn.linear_model import Lasso

    def L1(x_train, y_train, x_test, alpha):
        model = Lasso(alpha=alpha, random_state=0)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
#         y_fitted = pd.Series(model.predict(x_train), index=x_train.index)
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
    firm_list = df.index.get_level_values(0).unique()
    rolling_sample_size = 12
    test_periods = [(i, j) for i in [2018, 2019, 2020] for j in ["Q1", "Q2", "Q3", "Q4"]]
    
    y_test = pd.read_csv("../../assets/y_hats/univariate/y_test.csv", index_col=[0, 1, 2])
    
    # tune
    tune_space = np.linspace(0, 100, 101)
#     tune_space = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # LASSO
    t1 = time.time()
    y_hats = list(map(lambda firm: L1_i_tune(split_data_i(firm), tune_space), tqdm(firm_list)))
    t2 = time.time()
    print(t2-t1)
    
    y_hats = pd.concat(y_hats)
    y_hats.index = y_test.index
    y_hats.columns = ["y_hat_ml1_i_tuned"]    
    y_hats.to_csv("../../assets/y_hats/multivariate/y_hat_ml1_i_tuned.csv")
    y_hats = pd.read_csv("../../assets/y_hats/multivariate/y_hat_ml1_i_tuned.csv", index_col=[0, 1, 2])
    MAPEUB(y_test.values, y_hats.values)