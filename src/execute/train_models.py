# this is a kind of main execution file.
# Call SARIMA.py and nn.py, using moving_window to predict and output fitted model list and y_hat vectors.
# save trained models as pickle file.
if __name__ == "__main__":
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import pickle
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.sarima_multivariate import SARIMAModelList
    from utils.ts_split import train_test_split

    # read processed data
    df = pd.read_csv("../../data/processed/dataset.csv")
    
    # column names
    earning_v = df.columns[4: 10].values
    account_v_bs = df.columns[11:].values
    account_v_pl = df.columns[10:11].values
    
    # y
    y = df[earning_v[-1]]
    
    # x 
    x = df[np.append(account_v_bs, account_v_pl)]
    
    # time series train test split (4/5) : (1/5), yearly bases
    y_train, y_test = train_test_split(y, ratio=(4,1))
    x_train, x_test = train_test_split(x, ratio=(4,1))
    
    # Fit SARIMA
    
    # Brown & Rozeff
    # call SARIMAModelList class
    sarima_br = SARIMAModelList(order=(1, 0, 0), 
                          seasonal_order=(0, 1, 1, 4), 
                          y_prim_train=y_train,
                          y_test=y_test,
                          x_prim_train=x_train,
                          x_test=x_test,
                          silent=True
                         )
    y_hat_sarima_br = sarima_br.fit_rolling_window(len(y_train))
    y_hat_sarima_br.name = "y_hat_sarima_br"
    
    # save y_hat as csv file
    y_hat_sarima_br.to_csv('../../assets/y_hats/y_hat_sarima_br.csv')
    
    # save the class object as pickle file
    pickle.dump(sarima_br, open('../../assets/trained_models/sarima_br.pkl', 'wb'))
    # load pickle file
    #loaded_model = pickle.load(open('../../assets/trained_models/sarima_br.pkl', 'rb'))
    
    # Griffin
    sarima_g = SARIMAModelList(order=(0, 1, 1), 
                          seasonal_order=(0, 1, 1, 4), 
                          y_prim_train=y_train,
                          y_test=y_test,
                          x_prim_train=x_train,
                          x_test=x_test,
                          silent=True
                         )
    y_hat_sarima_g = sarima_g.fit_rolling_window(len(y_train))
    y_hat_sarima_g.name = "y_hat_sarima_g"
    y_hat_sarima_g.to_csv('../../assets/y_hats/y_hat_sarima_g.csv')
    pickle.dump(sarima_g, open('../../assets/trained_models/sarima_g.pkl', 'wb'))
    
    # Foster
    sarima_f = SARIMAModelList(order=(1, 0, 0), 
                          seasonal_order=(0, 1, 0, 4), 
                          y_prim_train=y_train,
                          y_test=y_test,
                          x_prim_train=x_train,
                          x_test=x_test,
                          silent=True
                         )
    y_hat_sarima_f = sarima_f.fit_rolling_window(len(y_train))
    y_hat_sarima_f.name = "y_hat_sarima_f"
    y_hat_sarima_f.to_csv('../../assets/y_hats/y_hat_sarima_f.csv')
    pickle.dump(sarima_f, open('../../assets/trained_models/sarima_f.pkl', 'wb'))