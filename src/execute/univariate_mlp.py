# Run NN-MLP
# this is some sort of main execution file.

if __name__ == "__main__":
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import torch
    import math
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.nn import MLP
    from utils.data_editor import lag, train_test_split
    
    # set seeds for reproductibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Prepare Data --> 関数
    
    # read processed data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    
    # empty list for dataframes
    y_test_list = []
    y_hat_umlp = []
    
    # For Loop firms and fit, predict
    i = df.index.get_level_values(0).unique()[0]
    
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
    
#     # time series train test split (4/5) : (1/5), yearly bases
#     target_train, target_test = train_test_split(target, ratio=(4,1))
#     feature_train, feature_test = train_test_split(feature, ratio=(4,1))
    
    # drop nan caused by lag()
    feature = feature.dropna(axis=0)
    target = target[feature.index]
    
    # setting torch
    dtype = torch.float # double float problem in layer 
    device = torch.device("cpu")
    
    # Make data to torch.tensor
    target = torch.tensor(target.values, dtype=dtype)
    feature = torch.tensor(feature.values, dtype=dtype)
    
    # rolling window data split
    train_window = 4 * 9 # all period: 48, train 36, test 12
    
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, feature, target, train_window):
            """
            Time Series torch Dataset class for rolling window process preparation
            
            Parameters
            ----------
            feature: 
                nD tensor X
            target: 
                1D tensor y
            train_window : int
                train periods for each rolling window
            
            Return
            ------
            len(target) - train_window sets of feature_train and target_train tensors (sibgel batch axis = 0)
            overview:
                ((batch_size=1, train_window, num_features), (batch_size=1, train_window))
                ...
                ((batch_size=1, train_window, num_features), (batch_size=1, train_window)). # len(target) - train_window
            """
            self.target = target
            self.feature = feature
            self.train_window = train_window
        
        def __len__(self):
            # num of output rolling window sets
            return len(self.target) - self.train_window
        
        def __getitem__(self, index):
            return (self.feature[index: index + self.train_window], self.target[index: index + self.train_window])
            
    
    ts_dataset = TimeSeriesDataset(feature, target, train_window)
    data_loader = torch.utils.data.DataLoader(ts_dataset, batch_size=1, shuffle=False)

    for i, d in enumerate(data_loader):
        print(i, d[0][0].size(), d[1][0].size()) # axis=0 はバッチサイズで1
    
    # Train MLP
    
    ### ! Hyper-Parameter ! ##########################################################
    hidden_units = 1000
    learning_rate = 1e-3
    num_iteration = 20000
    # Optimizer
    
    model_name = 'umlp' + '_hid' + str(hidden_units) + '_lr' + str(learning_rate) + '_iter' + str(num_iteration)
    ##################################################################################
    
    # Construct MLP class
        
    mlp = MLP(input_features=feature_train.size()[1], hidden_units=hidden_units, output_units=1)
    # print(mlp)

    # Save the pre_trained model
    # PATH = ''
    # torch.save(mlp.state_dict(), PATH)

    # Access to mlp weights
    # print(list(mlp.parameters())) # iteretor, just for printing
    # print(mlp.hidden.weight.size(), mlp.hidden.bias.size()) # editable ?
    # print(mlp.output.weight.size(), mlp.output.bias.size())

    # Construct loss and optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate) # link to mlp parameters (lr should be 1e-2)

    # unfold dimension to make our rolling window
    # ie. window size of 6, step size of 1
    target_train.unfold(0, 6, 1)
    
    # moving window 実装
    
    # Train the model: Learning iteration
    for step in range(num_iteration):
        # Forward pass
        target_pred = mlp(feature_train)
        # let y_pred be the same size as y
        target_pred = target_pred.squeeze(1)

        # Compute loss
        loss = criterion(target_pred, target_train) # link to mlp output
        if step % 1000 == 999:
            print(f"step {step}: loss {loss.item()}")

        # Zero gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # predict y_hat (target_hat) <- 良くないかも、with torch_nograd() と model.eval()
    y_hat_mlp_mv = mlp(feature_test).squeeze().detach().numpy()
    y_hat_mlp_mv = pd.Series(y_hat_mlp_mv)
    y_hat_mlp_mv.name = 'y_hat_mlp_mv'
    y_hat_mlp_mv.to_csv('../../assets/y_hats/y_hat_' + model_name + '.csv')
    
    # Save the pre_trained model
    PATH = '../../assets/trained_models/' + model_name + '.pth'
    torch.save(mlp.state_dict(), PATH)