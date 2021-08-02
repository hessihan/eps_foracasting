# Evaluate predicted value (y_hats) for each methods.

# build forecasting accuracy indicators table
def accuracy_table(y_test, y_hats, indicators):
    """
    
    Return a forecasting accuracy indicators table for each y_hats of forecasting methods.
    
    Parameters
    ----------
    y_test : pd.Series
        The true data
    y_hats : dict
        The predicted data. {"name" : np.array}
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
    for ind in indicators.keys():
        m = []
        # loop for each forecasting model
        for model in y_hats.keys():
            m.append(indicators[ind](y_test.values, y_hats[model].values))
        l.append(m)
    
    a = pd.DataFrame(l,
                     index=indicators.keys(),
                     columns=y_hats.keys()
                    )
    return a

def accuracy_table_i(y_test, y_hats, indicators):
    """
    
    Return a forecasting accuracy indicators table for each y_hats of forecasting methods
    for each individual firms.
    
    Parameters
    ----------
    y_test : pd.Series
        The true data
    y_hats : dict
        The predicted data. {"name" : np.array}
    indicators : dict
        The forecasting accuracy functions. {"name" : function}
        Each function is called from src/utils/accuracy.py

    Returns
    -------
    pd.DataFrame
        Forecasting accuracy indicators table.
    
    """
    l = []
    # loop for each forecasting model
    for model in y_hats.keys():
        m = []
        index_tuples = []
        # loop for each firm
        for firm in y_test.index.get_level_values(0).unique():
            # loop for each accuracy indicators
            for ind in indicators.keys():
                m.append(indicators[ind](
                    y_test.loc[pd.IndexSlice[firm, :, :], :].values, 
                    y_hats[model].loc[pd.IndexSlice[firm, :, :], :].values
                ))
                index_tuples.append((firm, ind))
        l.append(m)
    
    a = pd.DataFrame(l).T
    a.index = pd.MultiIndex.from_tuples(index_tuples)
    a.columns = y_hats.keys()
                    
    return a

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
    
    # read y_hat csv
#     dir_path = "./../../assets/y_hats/univariate/"
    dir_path = "./../../assets/y_hats/multivariate/"

    # get all paths in selected directody
    file_paths = os.listdir(dir_path)
    y_hats = []
    for path in file_paths:
        if path[:5] == "y_hat":
            # "y_hat"から始まるファイルのみ読み込み
            y_hats.append(pd.read_csv(dir_path + path, index_col=[0, 1, 2]))
    
    # cumpute forecast accuracy indicators
    # accuracy function dict
    ind_name = ["MAE", "MAPE", "MSE", "RMSE", "RMSPE"]
    indicators = [MAE, MAPE, MSE, RMSE, RMSPE]
    indicators = dict(zip(ind_name, indicators))
    
    # y_hats dictionary
#     method_name = ["SARIMA: BR",
#                    "SARIMA: G", 
#                    "SARIMA: F"
#                   ]
    method_name = ["MLM.1",
                   "MLM.2"
                  ]


    y_hats = dict(zip(method_name, y_hats))
    
    # accuracy table for all firm mean
    a = accuracy_table(y_test, y_hats, indicators)
#     a.to_csv("../../assets/y_hats/univariate/accuracy_table_u.csv")
    a.to_csv("../../assets/y_hats/multivariate/accuracy_table_m.csv")
    print(a)
    
    # accuracy table for each individual firms
    ai = accuracy_table_i(y_test, y_hats, indicators)
    ai.to_csv("../../assets/y_hats/multivariate/accuracy_table_mi.csv")
    print(ai)
    
#     # plot each y_hat series
#     fig = plt.figure(figsize=(16, 9))
#     ax = fig.add_subplot(111)
#     ax.plot(y_test, marker="o", label="y: record")
#     for i in y_hats.keys():
#         ax.plot(y_hats[i], marker="o", label=i, linestyle="--")
#     ax.legend()
#     plt.title("Plot y_hats for Test Data (Multivariate)")
# #     fig.savefig("../../assets/y_hats_plot_mv.png", format="png", dpi=300)
#     plt.show()
