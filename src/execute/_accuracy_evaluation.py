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
            m.append(indicators[ind](y_test, y_hats[model]))
        l.append(m)
    
    a = pd.DataFrame(l, 
                     index=indicators.keys(),
                     columns=y_hats.keys()
                    )
    return a

if __name__ == "__main__":
    # import external packages
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # import internal modules
    sys.path.insert(1, '../')
    from utils.accuracy import *
    
    # read y_test cdv
    y_test = np.loadtxt("../../assets/y_hats/y_test.csv", delimiter=',', skiprows=1, usecols=1)
    
    # read y_hat csv
    y_hats = [
        np.loadtxt("../../assets/y_hats/y_hat_sarima_br.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_sarima_g.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_sarima_f.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_mlp_mv_hid100_lr1en2_iter20k.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_mlp_mv_hid100_lr1en3_iter20k.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_mlp_mv_hid100_lr1en5_iter20k.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_mlp_mv_hid200_lr1en3_iter20k.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_mlp_mv_hid1k_lr1en3_iter20k.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_lstm_mv_hid100_lr0.005_epoch1000.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_lstm_mv_hid100_lr0.001_epoch1000.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_lstm_mv_hid200_lr0.001_epoch1000.csv", delimiter=',', skiprows=1, usecols=1),
        np.loadtxt("../../assets/y_hats/y_hat_lstm_mv_hid500_lr0.001_epoch1000.csv", delimiter=',', skiprows=1, usecols=1),
    ]
    # cumpute forecast accuracy indicators
    # accuracy function dict
    ind_name = ["MAE", "MAPE", "MSE", "RMSE", "RMSPE"]
    indicators = [MAE, MAPE, MSE, RMSE, RMSPE]
    indicators = dict(zip(ind_name, indicators))
    
    # y_hats dictionary
    method_name = ["SARIMA: BR", "SARIMA: G", "SARIMA: F", 
                   "MLP: hid 100, lr 1e-2, step 20k",
                   "MLP: hid 100, lr 1e-3, step 20k",
                   "MLP: hid 100, lr 1e-5, step 20k", 
                   "MLP: hid 200, lr 1e-3, step 20k", 
                   "MLP: hid 1k, lr 1e-3, step 20k",
                   "LSTM hid 100, lr 0.005, epoch 1k",
                   "LSTM hid 100, lr 0.001, epoch 1k",
                   "LSTM hid 200, lr 0.001, epoch 1k",
                   "LSTM hid 500, lr 0.001, epoch 1k",
                  ]

    y_hats = dict(zip(method_name, y_hats))
    
    a = accuracy_table(y_test, y_hats, indicators)
    a.to_csv("../../assets/accuracy_table_mv.csv")
    print(a)
    
    # plot each y_hat series
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.plot(y_test, marker="o", label="y: record")
    for i in y_hats.keys():
        ax.plot(y_hats[i], marker="o", label=i, linestyle="--")
    ax.legend()
    plt.title("Plot y_hats for Test Data (Multivariate)")
    fig.savefig("../../assets/y_hats_plot_mv.png", format="png", dpi=300)
    plt.show()
