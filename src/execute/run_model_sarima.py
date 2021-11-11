if __name__ == "__main__":
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import pickle
    from tqdm import tqdm
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.SARIMA import SARIMAModelList
    from utils.data_editor import train_test_split
    
    # Prepare Data
    
    # read processed data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    
    # empty list for dataframes
    y_test_list = []
    y_hat_sarima_br_list = []
    y_hat_sarima_g_list = []
    y_hat_sarima_f_list = []
    
#     # empty list for fitted class with fitted models for each windows 
#     # (memory ran out!!!)
#     fitted_model_sarima_br_list = []
#     fitted_model_sarima_g_list = []
#     fitted_model_sarima_f_list = []
    
    # For Loop firms and fit, predict SARIMA
    for i in tqdm(df.index.get_level_values(0).unique().values):
    
        # y : "EPS"
        y = df.loc[pd.IndexSlice[i, :, :], "EPS"]

        # x, exogenous regressors : 'INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF'
#         x = df.loc[pd.IndexSlice[i, :, :], ['INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF']]

        # time series train test split (4/5) : (1/5), yearly bases
        y_train, y_test = train_test_split(y, ratio=(4,1))
#         x_train, x_test = train_test_split(x, ratio=(4,1))

        # store y_test_i for calculating accuracy
        y_test_list.append(y_test)

        # Fit SARIMA

        # Brown & Rozeff (1, 0, 0) x (0, 1, 1)_4
        # call SARIMAModelList class
        sarima_br = SARIMAModelList(order=(1, 0, 0), 
                                    seasonal_order=(0, 1, 1, 4), 
                                    y_prim_train=y_train,
                                    y_test=y_test,
                                    multivariate=False,
                                    x_prim_train=None,
                                    x_test=None,
                                    silent=True
                                   )
        # fit SARIMA: rolling window size = len(y_train)
        y_hat_sarima_br = sarima_br.fit_rolling_window(len(y_train))
        y_hat_sarima_br_list.append(y_hat_sarima_br)
#         fitted_model_sarima_br_list.append(sarima_br)
        
        # Griffin (0, 1, 1) x (0, 1, 1)_4
        sarima_g = SARIMAModelList(order=(0, 1, 1), 
                                    seasonal_order=(0, 1, 1, 4), 
                                    y_prim_train=y_train,
                                    y_test=y_test,
                                    multivariate=False,
                                    x_prim_train=None,
                                    x_test=None,
                                    silent=True
                                   )
        y_hat_sarima_g = sarima_g.fit_rolling_window(len(y_train))
        y_hat_sarima_g_list.append(y_hat_sarima_g)
#         fitted_model_sarima_g_list.append(sarima_g)
        
        # Foster (1, 0, 0) x (0, 1, 0)_4
        sarima_f = SARIMAModelList(order=(1, 0, 0), 
                                    seasonal_order=(0, 1, 0, 4), 
                                    y_prim_train=y_train,
                                    y_test=y_test,
                                    multivariate=False,
                                    x_prim_train=None,
                                    x_test=None,
                                    silent=True
                                   )
        y_hat_sarima_f = sarima_f.fit_rolling_window(len(y_train))
        y_hat_sarima_f_list.append(y_hat_sarima_f)
#         fitted_model_sarima_f_list.append(sarima_f)

    y_test_series = pd.concat(y_test_list)
    y_test_series.name = "y_test"
    y_test_series.to_csv('../../assets/y_hats/univariate/y_test_.csv')
    
    y_hat_sarima_br_series = pd.concat(y_hat_sarima_br_list)
    y_hat_sarima_br_series.name = "y_hat_sarima_br"

    y_hat_sarima_g_series = pd.concat(y_hat_sarima_g_list)
    y_hat_sarima_g_series.name = "y_hat_sarima_g"

    y_hat_sarima_f_series = pd.concat(y_hat_sarima_f_list)
    y_hat_sarima_f_series.name = "y_hat_sarima_f"
    
    # save y_hat as csv file
    y_hat_sarima_br_series.to_csv('../../assets/y_hats/univariate/y_hat_sarima_br.csv')
    y_hat_sarima_g_series.to_csv('../../assets/y_hats/univariate/y_hat_sarima_g.csv')
    y_hat_sarima_f_series.to_csv('../../assets/y_hats/univariate/y_hat_sarima_f.csv')
    
#     # save the class object as pickle file
#     pickle.dump(fitted_model_sarima_br_list, open('../../assets/trained_models/univariate/sarima_br_u.pkl', 'wb'))  
#     pickle.dump(fitted_model_sarima_g_list, open('../../assets/trained_models/univariate/sarima_g_u.pkl', 'wb'))  
#     pickle.dump(fitted_model_sarima_f_list, open('../../assets/trained_models/univariate/sarima_f_u.pkl', 'wb'))  

    # load pickle file
#     loaded_model = pickle.load(open('../../assets/trained_models/sarima_br.pkl', 'rb'))