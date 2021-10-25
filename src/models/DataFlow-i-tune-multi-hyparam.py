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

def tune_i(df, firm_list, test_periods, val_size, i, method, tune_space):
    """
    df: 
        master dataframe
    firm_list
    method: func
        model function method(x_train, y_train, x_test, hyparams) which returns y_hat 
    tune space: np.meshgrid
        grid to search hyper parameter
    """
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

    def split_data_i(df, firm_list, test_periods, val_size, i):        
        splitted_data_i = {
            "x_test": [], "y_test": [],
            "x_train_val": [], "y_train_val": [], 
            "x_val": [], "y_val": [], 
            "x_train": [], "y_train": []
        }
        for j in test_periods:
            df_ij = dataslice(df, firm_list, test_periods, val_size, i, j)
            splitted_data_i["x_test"].append(df_ij["x_test"])
            splitted_data_i["y_test"].append(df_ij["y_test"])
            splitted_data_i["x_train_val"].append(df_ij["x_train_val"])
            splitted_data_i["y_train_val"].append(df_ij["y_train_val"])
            splitted_data_i["x_val"].append(df_ij["x_val"])
            splitted_data_i["y_val"].append(df_ij["y_val"])
            splitted_data_i["x_train"].append(df_ij["x_train"])
            splitted_data_i["y_train"].append(df_ij["y_train"])
        return splitted_data_i

    def prim_train(hyparam):
        y_val_hat = np.arange(len(test_periods))
        y_val_hat = list(map(lambda period: method(
            *hyparam,
            splitted_data_i["x_train"][period], 
            splitted_data_i["y_train"][period], 
            splitted_data_i["x_val"][period]
        ), y_val_hat))
        y_val_hat = pd.concat(y_val_hat)
        return y_val_hat

    def search_best_hyparam(tune_space):
        y_val = pd.concat(splitted_data_i["y_val"])
        val_mspe = np.array(list(map(lambda hyparam: MSE(y_val, prim_train(hyparam)), tune_space)))
        best_hyparam = tune_space[val_mspe.argmin()]
#         print("best hyparam:", best_hyparam)
#         print("min val MSPE:", val_mspe.min())
        return best_hyparam

    def final_train(best_hyparam):
        y_hat = np.arange(len(test_periods))
        y_hat = list(map(lambda period: method(
            *best_hyparam,
            splitted_data_i["x_train_val"][period], 
            splitted_data_i["y_train_val"][period], 
            splitted_data_i["x_test"][period]
        ), y_hat))
        y_hat = pd.concat(y_hat)
        return y_hat
    
    splitted_data_i = split_data_i(df, firm_list, test_periods, val_size, i)
    best_hyparam = search_best_hyparam(tune_space)
    y_hat = final_train(best_hyparam)
    return y_hat
#######################################################################

# Debug
if __name__ == "__main__":
    
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from multiprocessing import Pool, cpu_count

    def L1(alpha, x_train, y_train, x_test):
        model = Lasso(alpha=alpha, random_state=0)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
        return y_hat

    def L2(alpha, x_train, y_train, x_test):
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
        return y_hat
    
    def EN(hyparams_1, hyparams_2, x_train, y_train, x_test):
        model = ElasticNet(alpha=hyparams_1, l1_ratio=hyparams_2)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
        return y_hat    

    def RAF(hyparams_1, hyparams_2, x_train, y_train, x_test):  
        model = RandomForestRegressor(n_estimators=hyparams_1, max_depth=hyparams_2, max_features="auto", random_state=0)
        model.fit(x_train, y_train)
        y_hat = pd.Series(model.predict(x_test), index=x_test.index)
        return y_hat    
    
    # PREPARE LAGGED DATA
    my_df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    col = ['EPS', 'INV', 'AR', 'CAPX','GM', 'SA', 'ETR', 'LF'] # multivariate
#     col = ['EPS'] # univariate
    my_df = my_df[col]
    my_df = pd.concat([
        my_df,
        lagged_data(my_df, col, 1),
        lagged_data(my_df, col, 2),
        lagged_data(my_df, col, 3),
        lagged_data(my_df, col, 4),
    ], axis=1)
    my_df.dropna(inplace=True)
    my_firm_list = my_df.index.get_level_values(0).unique()
    my_test_periods = [(i, j) for i in [2018, 2019, 2020] for j in ["Q1", "Q2", "Q3", "Q4"]]
    
    y_test = pd.read_csv("../../assets/y_hats/univariate/y_test.csv", index_col=[0, 1, 2])
    
    # TUNING
#     my_tune_space = np.linspace(0, 100, 101)
#     my_tune_space = np.meshgrid([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    
    # single
#     my_firm_list = my_firm_list[:2]
#     t1 = time.time()
#     y_hats = list(map(lambda firm: tune_i(my_df, my_firm_list, my_test_periods, 1, firm, 
#                                           L1, my_tune_space), tqdm(my_firm_list)))
#     t2 = time.time()
#     print(t2-t1)
    
    # Multiprocessing
#     https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function
    class Tuner_i(object):
        def __init__(self, df, firm_list, test_periods, val_size, method, tune_space):
            self.df = df
            self.firm_list = firm_list
            self.test_periods = test_periods
            self.val_size = val_size
            self.method = method
            self.tune_space = tune_space
        def __call__(self, i):
            return tune_i(self.df, self.firm_list, self.test_periods, self.val_size, i, self.method, self.tune_space)
    
    # LASSO
#     my_firm_list = my_firm_list[:2]
    my_tune_space = [[0.001], [0.01], [0.1], [1], [10], [100], [1000]]
    t1 = time.time()
    p = Pool(cpu_count() - 1)
    y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, L1, my_tune_space), tqdm(my_firm_list)))
    p.close()
    t2 = time.time()
    print(t2-t1)
    
    name = #"y_hat_ml1_i_tuned_simple"
    y_hats = pd.concat(y_hats)
    y_hats.index = y_test.index
    y_hats.name = name
    y_hats.to_csv("../../assets/y_hats/multivariate/" + name + ".csv")
    y_hats = pd.read_csv("../../assets/y_hats/multivariate/" + name + ".csv", index_col=[0, 1, 2])
    MAPEUB(y_test.values, y_hats.values)
    
    # Ridge
    my_tune_space = [[0.001], [0.01], [0.1], [1], [10], [100], [1000]]
    t1 = time.time()
    p = Pool(cpu_count() - 1)
    y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, L2, my_tune_space), tqdm(my_firm_list)))
    p.close()
    t2 = time.time()
    print(t2-t1)
    
    name = #"y_hat_ml2_i_tuned_simple"
    y_hats = pd.concat(y_hats)
    y_hats.index = y_test.index
    y_hats.name = name
    y_hats.to_csv("../../assets/y_hats/multivariate/" + name + ".csv")
    y_hats = pd.read_csv("../../assets/y_hats/multivariate/" + name + ".csv", index_col=[0, 1, 2])
    MAPEUB(y_test.values, y_hats.values)
    
    # EN
    my_tune_space = np.vstack(map(np.ravel, np.meshgrid(
        [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ))).T
    t1 = time.time()
    p = Pool(cpu_count() - 1)
    y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, EN, my_tune_space), tqdm(my_firm_list)))
    p.close()
    t2 = time.time()
    print(t2-t1)
    
    name = "y_hat_men_i_tuned_simple"
    y_hats = pd.concat(y_hats)
    y_hats.index = y_test.index
    y_hats.name = name
    y_hats.to_csv("../../assets/y_hats/multivariate/" + name + ".csv")
    y_hats = pd.read_csv("../../assets/y_hats/multivariate/" + name + ".csv", index_col=[0, 1, 2])
    MAPEUB(y_test.values, y_hats.values)

    # RAF
    my_tune_space = np.vstack(map(np.ravel, np.meshgrid(
        [100, 500, 1000, 2000],
        [10, 100, None]
    ))).T
    t1 = time.time()
    p = Pool(4) #Pool(cpu_count() - 1)
    y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, RAF, my_tune_space), tqdm(my_firm_list)))
    p.close()
    t2 = time.time()
    print(t2-t1)
    
    name = "y_hat_mraf_i_tuned_simple_"
    y_hats = pd.concat(y_hats)
    y_hats.index = y_test.index
    y_hats.name = name
    y_hats.to_csv("../../assets/y_hats/multivariate/" + name + ".csv")
    y_hats = pd.read_csv("../../assets/y_hats/multivariate/" + name + ".csv", index_col=[0, 1, 2])
    MAPEUB(y_test.values, y_hats.values)
