if __name__ == "__main__":
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import pickle
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.MLM import MLMModelList
    from utils.data_editor import train_test_split
    
    # Prepare Data
    
    # read processed data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    
    # empty list for dataframes
    y_test_list = []
    y_hat_mlm1_list = []
    y_hat_mlm2_list = []
    
#     # empty list for fitted class with fitted models for each windows 
#     # (memory ran out!!!)
#     fitted_model_mlm1_list = []
#     fitted_model_mlm2_list = []
    
    # For Loop firms and fit, predict SARIMA
    for i in df.index.get_level_values(0).unique().values:
    
        # y : "EPS"
        y = df.loc[pd.IndexSlice[i, :, :], "EPS"]

        # x, exogenous regressors : 'INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF'
        x = df.loc[pd.IndexSlice[i, :, :], ['INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF']]

        # time series train test split (4/5) : (1/5), yearly bases
        y_train, y_test = train_test_split(y, ratio=(4,1))
        x_train, x_test = train_test_split(x, ratio=(4,1))

        # とりあえずザックリデータ分割。yのlagやxのlagはMLMModelList内でやる。
        
        # store y_test_i for calculating accuracy
        y_test_list.append(y_test)
        
        # Fit MLM
        
        # MLM.1 (lag1 X)
        mlm1 = MLMModelList(
            y_prim_train=y_train,
            y_test=y_test,
            x_lag=1,
            x_prim_train=x_train,
            x_test=x_test, 
            silent=True,
            store_models=False
        )
        
        # fit MLM.1: rolling window size = len(y_train) - 4 (max lag size)
        y_hat_mlm1 = mlm1.fit_rolling_window(window=len(y_train)-4)
        y_hat_mlm1_list.append(y_hat_mlm1)
#         fitted_model_mlm1_list.append(mlm1)
        
        # MLM.2 (lag4 X)
        mlm2 = MLMModelList(
            y_prim_train=y_train,
            y_test=y_test,
            x_lag=4,
            x_prim_train=x_train,
            x_test=x_test, 
            silent=True,
            store_models=False
        )
        
        # fit MLM.2: rolling window size = len(y_train) - 4 (max lag size)
        y_hat_mlm2 = mlm2.fit_rolling_window(window=len(y_train)-4)
        y_hat_mlm2_list.append(y_hat_mlm2)
#         fitted_model_mlm2_list.append(mlm2)

        print(i, " Done")

    y_test_series = pd.concat(y_test_list)
    y_test_series.name = "y_test"
    y_test_series.to_csv('../../assets/y_hats/multivariate/y_test.csv')
    
    y_hat_mlm1_series = pd.concat(y_hat_mlm1_list)
    y_hat_mlm1_series.name = "y_hat_mlm1"

    y_hat_mlm2_series = pd.concat(y_hat_mlm2_list)
    y_hat_mlm2_series.name = "y_hat_mlm2"
    
    # save y_hat as csv file
    y_hat_mlm1_series.to_csv('../../assets/y_hats/multivariate/y_hat_mlm1.csv')
    y_hat_mlm2_series.to_csv('../../assets/y_hats/multivariate/y_hat_mlm2.csv')
    
#     # save the class object as pickle file
#     pickle.dump(fitted_model_mlm1_list, open('../../assets/trained_models/multivariate/mlm1.pkl', 'wb'))  
#     pickle.dump(fitted_model_mlm2_list, open('../../assets/trained_models/multivariate/mlm2.pkl', 'wb'))  

    # load pickle file
#     loaded_model = pickle.load(open('../../assets/trained_models/multivariate/mlm1.pkl', 'rb'))