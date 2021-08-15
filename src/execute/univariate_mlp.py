# Run NN-MLP
# this is some sort of main execution file.

if __name__ == "__main__":
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import torch
    import math
    import time
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.nn import MLP, TimeSeriesDataset
    from utils.data_editor import lag, train_test_split
    
    # set seeds for reproductibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Prepare Data --> 関数化させる?
    
    # read processed data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    
    # empty list for dataframes
    y_test_list = []
    y_hat_umlp = []
    
    t1 = time.time() 
    
    # For Loop firms and fit, predict
    for i in df.index.get_level_values(0).unique():
        print(i)

        # y : "EPS"
        y = df.loc[pd.IndexSlice[i, :, :], "EPS"]

        # x, exogenous regressors : 'INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF'
    #     x = df.loc[pd.IndexSlice[i, :, :], ['INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF']]

        # Unlike statsmodel SARIMA package, NN needs to prepare lagged inputs manually if needed.
        # y_lag and x_lag (lag 4 for now)
        num_lag = 4
        y_lag = lag(y, num_lag, drop_nan=False, reset_index=False)
    #     x_lag = lag(x, num_lag, drop_nan=False, reset_index=False)

        # Redefine data name as target (y) and feature (y_lag) (explanatory variable, predictor)
        target = y
        feature = y_lag

        # save simple test data series
        _, target_test_dataset = train_test_split(target, ratio=(4,1))
        _, feature_test_dataset = train_test_split(feature, ratio=(4,1))

        # drop nan caused by lag()
        feature = feature.dropna(axis=0)
        target = target[feature.index]

        # setting torch
        dtype = torch.float # double float problem in layer 
        device = torch.device("cpu")

        # Make data to torch.tensor
        target = torch.tensor(target.values, dtype=dtype)
        feature = torch.tensor(feature.values, dtype=dtype)
        target_test_dataset = torch.tensor(target_test_dataset.values, dtype=dtype)
        feature_test_dataset = torch.tensor(feature_test_dataset.values, dtype=dtype)

        # rolling window data preparation
        ### ! Hyper-Parameter ! ##########################################################
        train_window = 4 * 9 # all period: 48, train 36, test 12
        ##################################################################################

        rolling_window_train_dataset = TimeSeriesDataset(feature, target, train_window)
    #     len(train_dataset) == len(target) - train_window

        rolling_window_train_loader = torch.utils.data.DataLoader(rolling_window_train_dataset, batch_size=1, shuffle=False)

        # check rolling window data flow
    #     for i, (feature_train, target_train) in enumerate(rolling_window_train_loader):
    #         print(i, feature_train.size(), target_train.size()) # axis=0 はバッチサイズで1
    
        # Train MLP

        ### ! Hyper-Parameter ! ##########################################################
        hidden_units = 1000
        learning_rate = 0.001
        num_iteration = 10000
        # Optimizer

        model_name = 'umlp' + '_hid' + str(hidden_units) + '_lr' + str(learning_rate) + '_iter' + str(num_iteration)
        ##################################################################################
        
        # load rolling window data flow
        for num_window, (feature_train, target_train) in enumerate(rolling_window_train_loader):
            print("rolling window: ", num_window)
            feature_train = feature_train[0] # extract single batch
            target_train = target_train[0] # extract single batch
            
            #(only first window)
            if num_window == 0:
                # Construct MLP class 
                mlp = MLP(input_features=feature_train.size()[1], hidden_units=hidden_units, output_units=1)

                # Construct loss and optimizer
                criterion = torch.nn.MSELoss(reduction="mean")
                optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate) # link to mlp parameters (lr should be 1e-2)
            else:
                pass
            
            # Train the model: Learning iteration
            # use pre window trained model's weight as initial weight (continue training)
#             print("initial weight: ", mlp.state_dict()['hidden.weight'][0])
            
            for step in range(num_iteration):
                # Forward pass
                target_pred = mlp(feature_train)
                # let y_pred be the same size as y
                target_pred = target_pred.squeeze(1)

                # Compute loss
                loss = criterion(target_pred, target_train) # link to mlp output
                if (step == 0) | (step == 5000) | (step == 9999):
                    print(f"step {step}: loss {loss.item()}")

                # Zero gradients, perform backward pass, and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Save the trained model
            PATH = '../../assets/trained_models/univariate/mlp/' + model_name + "_" + i + "_" + "win" + str(num_window) + '.pth'
            torch.save(mlp.state_dict(), PATH)
            # use the existing trained model and continute training next window.
#             print("inherit weight: ", mlp.state_dict()['hidden.weight'][0])
            
            # predict y_hat (target_hat) <- 良くないかも、with torch_nograd() と model.eval()
            with torch.no_grad():
                target_test = target_test_dataset[num_window]
                feature_test = feature_test_dataset[num_window]
                y_hat_umlp.append(mlp(feature_test).squeeze().detach().numpy())
#                 print(feature_test)
#                 print(target_test)
#                 print(y_hat_umlp[-1])
        
    y_hat_umlp = pd.Series(y_hat_umlp)
    y_hat_umlp.index = df.loc[pd.IndexSlice[:, 2018:, :], :].index
    y_hat_umlp.name = 'y_hat_umlp'
    y_hat_umlp.to_csv('../../assets/y_hats/univariate/y_hat_' + model_name + '.csv')
    
    t2 = time.time()
    
    elapsed_time = t2-t1
    print(f"elapsedtime：{elapsed_time}")