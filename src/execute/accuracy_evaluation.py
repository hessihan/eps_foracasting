# Evaluate predicted value (y_hats) for each methods.
# import external packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# import internal modules
# sys.path.insert(1, '../')
sys.path.insert(1, "/mnt/d/0ngoing/thesis/repo/src/")
from utils.accuracy import *

# build forecasting accuracy indicators table
def accuracy_table(y_test, y_hats, indicators):
    """
    
    Return a forecasting accuracy indicators table for each y_hats of forecasting methods.
    
    Parameters
    ----------
    y_test : pd.Series
        The true data
    y_hats : pd.DataFrame
        The predicted data
    indicators : dict
        The forecasting accuracy functions. {"name" : function}
        Each function is called from src/utils/accuracy.py

    Returns
    -------
    pd.DataFrame
        Forecasting accuracy indicators table.
    
    """
    l = []
    # loop for each accuracy indicators
    for ind in indicators:
        m = []
        # loop for each forecasting model
        for model in y_hats.columns:
            m.append(indicators[ind](y_test.values, y_hats[model].values))
        l.append(m)
    
    a = pd.DataFrame(l,
                     index=indicators,
                     columns=y_hats.columns
                    )
    return a

def accuracy_table_i(y_test, y_hats_all, indicators):
    """
    
    Return a forecasting accuracy indicators table for each y_hats of forecasting methods
    for each individual firms.
    
    Parameters
    ----------
    y_test : pd.Series
        The true data
    y_hats_all : pd.DataFrame
        The predicted data for all firms
    indicators : dict
        The forecasting accuracy functions. {"name" : function}
        Each function is called from src/utils/accuracy.py

    Returns
    -------
    pd.DataFrame
        Forecasting accuracy indicators table.
    
    """
    ai = pd.DataFrame()
    for firm in y_test.index.get_level_values(0).unique():
        a = accuracy_table(y_test.loc[firm],
                           y_hats_all.loc[firm],
                           indicators
                          )
        a.index = pd.MultiIndex.from_tuples([(firm, i) for i in a.index])
        ai = pd.concat([ai, a], axis=0)
    ai
                    
    return ai

if __name__ == "__main__":
    # read y_test cdv
    y_test = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2]).loc[pd.IndexSlice[get_non_late_firm(), [2018, 2019, 2020], :], "EPS"]
    y_test.name = "y_test"
    y_test.to_csv("../../assets/y_hats/univariate/y_test.csv")
    
    ## make y_hats_all dataframe
    
    # Random walk prediction
    y_hat_rw = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2]).groupby(level=[0])['EPS'].shift(1).loc[y_test.index]
    y_hat_rw.name = "y_hat_rw"
    y_hat_rw.to_csv("./../../assets/y_hats/univariate/y_hat_rw.csv")

    # Seasonal Random walk prediction
    y_hat_srw = pd.DataFrame(pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2]))["EPS"]
    y_hat_srw = y_hat_srw.groupby(level=[0, 2]).shift(1)
    # y_hat_srw = y_hat_srw.groupby(level=[0]).shift(4)
    y_hat_srw = y_hat_srw.loc[y_test.index]
    y_hat_srw.name = "y_hat_srw"
    y_hat_srw.to_csv("./../../assets/y_hats/univariate/y_hat_srw.csv")
    
    y_hats_all = pd.DataFrame(y_test)
    # read y_hat csv
    
    # univariate
    dir_path = "./../../assets/y_hats/univariate/"

    # get all paths in selected directody
    file_paths = os.listdir(dir_path)
    for path in file_paths:
        if path[:5] == "y_hat":
            # "y_hat"から始まるファイルのみ読み込み
            y_hat = pd.read_csv(dir_path + path, index_col=[0, 1, 2]).loc[y_test.index].iloc[:, 0]
            y_hats_all = pd.concat([y_hats_all, y_hat], axis=1)

    # multivariate
    dir_path = "./../../assets/y_hats/multivariate/"

    # get all paths in selected directody
    file_paths = os.listdir(dir_path)
    for path in file_paths:
        if path[:5] == "y_hat":
            y_hat = pd.read_csv(dir_path + path, index_col=[0, 1, 2]).loc[y_test.index].iloc[:, 0]
            y_hats_all = pd.concat([y_hats_all, y_hat], axis=1)

    y_hats_all

    # Forecast combination
    mml =[
        'y_hat_ml1_i_tuned_simple',
        'y_hat_ml2_i_tuned_simple',
        'y_hat_men_i_tuned_simple',
        'y_hat_mraf_i_tuned_simple',
        'y_hat_mmlp'
        ]

    y_hats_all["y_hat_ALL_comb"] = y_hats_all.mean(axis=1)
    y_hats_all["y_hat_MML_comb"] = y_hats_all[mml].mean(axis=1)

    #################################################################################################### eliminate late firm
    # y_hats_all = y_hats_all.loc[get_non_late_firm()]
    ####################################################################################################
    
    # y_hats_all.to_csv("./../../assets/y_hats/y_hats_all.csv")

    ## agregate forceast accuracy score
    
    # forecast accuracy indicators
    ind_name = ["Max_error", "Max_percentage_error", "MAE", "MAPE", "MSPE", "MAPE-UB", "MSPE-UB", "Large_error_rate"]
    indicators = [Max_error, Max_percentage_error, MAE, MAPE, MSE, MAPEUB, MSPEUB, LargeErrorRate]
    indicators = dict(zip(ind_name, indicators))

    print("Num firm: ", y_hats_all.index.get_level_values(0).unique().shape)
    
    # accuracy table for all firm mean
    a = accuracy_table(y_hats_all["y_test"], y_hats_all, indicators).T
    a.to_csv("../../assets/y_hats/accuracy_table.csv")

    # accuracy table for all firm mean, by quarter

    
    # accuracy table for each individual firms
    ai = accuracy_table_i(y_hats_all["y_test"], y_hats_all, indicators)
    ai.to_csv("../../assets/y_hats/accuracy_table_i.csv")

    # primal accuracy table for paper
    model_list = [
        'y_hat_rw', 
        'y_hat_srw', 
        'y_hat_sarima_f', 
        'y_hat_sarima_g', 
        'y_hat_sarima_br',
        'y_hat_mlm1', 
        'y_hat_mlm2',
        'y_hat_mlm4',
        'y_hat_ul1_i_tuned_simple',
        'y_hat_ul2_i_tuned_simple',
        'y_hat_uen_i_tuned_simple',        
        'y_hat_umlp',
        'y_hat_ml1_i_tuned_simple',
        'y_hat_ml2_i_tuned_simple',
        'y_hat_men_i_tuned_simple',
        'y_hat_mraf_i_tuned_simple',
        'y_hat_mmlp'
        ]
    y_hat_list = list(map(lambda x: y_hats_all[x], model_list))
    q_list = ["Q1", "Q2", "Q3", "Q4", ["Q1", "Q2", "Q3", "Q4"]]
    score_list = [MAE, MAPEUB, MSPEUB, LargeErrorRate]    

    a_by_q = []
    for y_hat in y_hat_list:
        by_q = []
        for q in q_list:
            by_q.append(list(map(lambda s: s(y_hats_all["y_test"].loc[pd.IndexSlice[:, :, q]], y_hat.loc[pd.IndexSlice[:, :, q]]), score_list)))
        a_by_q.append(np.array(by_q).flatten())

    a_by_q = pd.DataFrame(a_by_q)
    a_by_q.index = model_list

    col = [(i, j) for i in ["Q1", "Q2", "Q3", "Q4", "Overall"] for j in ["MAE", "MAPE", "MSPE", "Large Forecast Error"]]
    a_by_q.columns = pd.MultiIndex.from_tuples(col)
    a_by_q.to_csv("../../assets/y_hats/accuracy_table_by_quarter.csv")

    # two year test range
    accuracy_table(y_test.loc[pd.IndexSlice[:, [2018, 2019], :]], y_hats_all.loc[pd.IndexSlice[:, [2018, 2019], :]], indicators).T