# Evaluate predicted value (y_hats) for each methods.

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
    # import external packages
    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # import internal modules
    sys.path.insert(1, '../')
    from utils.accuracy import *
    
    # read y_test cdv
    y_test = pd.read_csv("../../assets/y_hats/univariate/y_test.csv", index_col=[0, 1, 2])
    y_test = y_test[y_test.columns[0]] # to series
    
    ## make y_hats_all dataframe
    
    # Random walk prediction
    y_hat_rw = pd.DataFrame(pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2]).groupby(level=[0])['EPS'].shift(1).loc[y_test.index])
    y_hat_rw.columns = ["y_hat_rw"]
    y_hat_rw.to_csv("./../../assets/y_hats/univariate/y_hat_rw.csv")
    
    y_hats_all = pd.DataFrame(y_test)
    
    # read y_hat csv
    
    # univariate
    dir_path = "./../../assets/y_hats/univariate/"

    # get all paths in selected directody
    file_paths = os.listdir(dir_path)
    for path in file_paths:
        if path[:5] == "y_hat":
            # "y_hat"から始まるファイルのみ読み込み
            y_hats_all = pd.concat([y_hats_all, pd.read_csv(dir_path + path, index_col=[0, 1, 2])], axis=1)

    # multivariate
    dir_path = "./../../assets/y_hats/multivariate/"

    # get all paths in selected directody
    file_paths = os.listdir(dir_path)
    for path in file_paths:
        if path[:5] == "y_hat":
            # "y_hat"から始まるファイルのみ読み込み
            y_hats_all = pd.concat([y_hats_all, pd.read_csv(dir_path + path, index_col=[0, 1, 2])], axis=1)
    
    y_hats_all.to_csv("./../../assets/y_hats/y_hats_all.csv")
    
    ## sample specific (firm x quarter) error table
    
    # exact error
    error = []
    for i in y_hats_all.columns:
        error.append(y_test - y_hats_all[i])
    error = pd.DataFrame(error).T
    error.columns = ["y_test - " + i for i in  y_hats_all.columns]
    error.to_csv("../../assets/y_hats/error.csv")
    
    # absolute error
    error_abs = abs(error)
    error_abs.columns = ["|" + i + "|" for i in error.columns]
    error_abs.to_csv("../../assets/y_hats/error_abs.csv")
    
    # percentage error
    error_p = []
    for i in error.columns:
        error_p.append(error[i] / y_test)
    error_p = pd.DataFrame(error_p).T
    error_p.columns = ["(" + i + ") / y_test" for i in error.columns]
    error_p.to_csv("../../assets/y_hats/error_p.csv")
    
    # absolute percentage error
    error_p_abs = abs(error_p)
    error_p_abs.columns = ["|" + i + "|" for i in error_p.columns]
    error_p_abs.to_csv("../../assets/y_hats/error_p_abs.csv")
    
    ## agregate forceast accuracy score
    
    # forecast accuracy indicators
    ind_name = ["Max_error", "Max_percentage_error", "MAE", "MAPE", "MSE", "RMSE", "RMSPE"]
    indicators = [Max_error, Max_percentage_error, MAE, MAPE, MSE, RMSE, RMSPE]
    indicators = dict(zip(ind_name, indicators))
    
    # accuracy table for all firm mean
    a = accuracy_table(y_test, y_hats_all, indicators)
    a.to_csv("../../assets/y_hats/accuracy_table.csv")
    print(a)
    
    # accuracy table for each individual firms
    ai = accuracy_table_i(y_test, y_hats_all, indicators)
    ai.to_csv("../../assets/y_hats/accuracy_table_i.csv")
    print(ai)
    
    ## Upper bound |(Y_t - \hat Y_t) / Y_t| = 1 if exceed 1
    
    # absolute percentage error
    error_p_abs_ub = error_p_abs.copy()
    error_p_abs_ub[error_p_abs_ub > 1] = 1
    error_p_abs_ub.to_csv("../../assets/y_hats/error_p_abs_ub.csv")
    
#     error_p_sq_ub = error_p.copy()
#     error_p_sq_ub[abs(error_p_sq_ub) > 1] = 1
#     error_p_sq_ub = error_p_sq_ub ** 2
    
    # squared percentage error
    error_p_sq_ub = error_p_abs_ub ** 2
    error_p_sq_ub.to_csv("../../assets/y_hats/error_p_abs_ub.csv")
    
    # large error
    (error_p_abs_ub == 1).sum()
    (error_p_abs_ub == 1).sum() / len(error_p_abs_ub)
    
        # i
    (error_p_abs_ub == 1).sum(level=0)
    
    # aggregate table
    accuracy_table_ub = pd.DataFrame([error_p_abs_ub.mean(), 
                                      error_p_sq_ub.mean(), 
                                      (error_p_abs_ub == 1).sum() / len(error_p_abs_ub)
                                     ])
    accuracy_table_ub.index = ["MAPE", "MSPE", "Large Forecast Error"]
    accuracy_table_ub.columns = y_hats_all.columns
    accuracy_table_ub = accuracy_table_ub.T
    accuracy_table_ub.to_csv("../../assets/y_hats/accuracy_table_ub.csv")
    
        # i
    error_p_abs_ub.mean(level=0)