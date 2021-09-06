# Run NN-MLP
# this is some sort of main execution file.

# import external libraries
import sys
import numpy as np
import pandas as pd
import torch
import time

# import internal modules
sys.path.insert(1, '../')
from models.nn import MLP, TimeSeriesDataset
from utils.data_editor import lag, train_test_split
from utils.accuracy import MAE, MAPE, MSE, Error, AbsoluteError, PercentageError, AbsolutePercentageError

# set seeds for reproductibility
np.random.seed(0)
torch.manual_seed(0)

# ------------
# Prepare Data
# ------------

# read processed data
df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])

# empty list for dataframes
y_test_list = []
y_hat_umlp = []

t1 = time.time() 

# ------------
# Loop Firm
# ------------

# For Loop firms and fit, predict
# firm_list = df.index.get_level_values(0).unique()[:2]
firm_list = [df.index.get_level_values(0).unique()[0]]
for firm in firm_list:
    print(firm)

    # y : "EPS"
    y = df.loc[pd.IndexSlice[firm, :, :], "EPS"]

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
    rolling_sample_size = 12
    _, target_test_all = train_test_split(target, test_size=rolling_sample_size)
    _, feature_test_all = train_test_split(feature, test_size=rolling_sample_size)
    y_test_list.append(target_test_all)

    # drop nan caused by lag()
    feature = feature.dropna(axis=0)
    target = target[feature.index]

    # setting torch
    dtype = torch.float # double float problem in layer 
    device = torch.device("cpu")

    # Make data to torch.tensor
    target_all = torch.tensor(target.values, dtype=dtype)
    feature_all = torch.tensor(feature.values, dtype=dtype)
    target_test_all = torch.tensor(target_test_all.values, dtype=dtype)
    feature_test_all = torch.tensor(feature_test_all.values, dtype=dtype)

    # window size: train_val_window_size = 36, train_window_size = 24, val_window_size = 12, test_window_size = 1
    test_window_size = 1
    train_val_window_size = len(target_all) - rolling_sample_size
    val_window_size = 12
    train_window_size = train_val_window_size - val_window_size

    train_val_all_dataset = TimeSeriesDataset(feature_all, target_all, train_window=train_val_window_size)

    train_val_rolling_loader = torch.utils.data.DataLoader(train_val_all_dataset, batch_size=1, shuffle=False)

    # check rolling window data flow
    # for rolling_sample, (feature_train_val, target_train_val) in enumerate(train_val_rolling_loader):
    #     print(rolling_sample, feature_train_val, target_train_val)
    #     print(rolling_sample, feature_train_val.size(), target_train_val.size()) # axis=0 はバッチサイズで1

    # ------------
    # Train Model
    # ------------

    ### ! Hyper-Parameter ! ##########################################################
    hidden_units = 1000
    learning_rate = 1e-3
    num_epochs = 10000
    # Optimizer

    model_name = 'umlp' + '_hid' + str(hidden_units) + '_lr' + str(learning_rate) + '_epoch' + str(num_epochs)
    ##################################################################################

    # --------------------
    # Loop Rolling Sample
    # --------------------

    # load rolling window data flow
    for rolling_sample, (feature_train_val, target_train_val) in enumerate(train_val_rolling_loader):
        print("Firm: ", firm, "Rolling Window: ", rolling_sample)
        #===============
        # Data
        #===============
        feature_train_val = feature_train_val[0] # extract single rolling sample batch
        target_train_val = target_train_val[0] # extract single rolling sample batch

        # all "batch" dataset (unit window)
        train_val_dataset = TimeSeriesDataset(feature_train_val, target_train_val, train_window=None)
#         print(len(train_val_dataset))
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=len(train_val_dataset), shuffle=False)

        # 分け方は一応 full-psuedo で val 12, train 24でsplit。(val に rolling windowはしない)
        # batch はとりあえずなしで。
        train_dataset = torch.utils.data.dataset.Subset(train_val_dataset, list(range(0, train_window_size)))
#         print(len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

        val_dataset = torch.utils.data.dataset.Subset(train_val_dataset, list(range(train_window_size, len(train_val_dataset))))
#         print(len(val_dataset))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        test_dataset = TimeSeriesDataset(feature_test_all[rolling_sample].reshape(1, -1), target_test_all[rolling_sample].reshape(-1), train_window=None)
#         print(len(test_dataset))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        #===============
        # Train Val Fit
        #===============
        #(initiate with random init params for only first rolling sample)
        if rolling_sample == 0:
            # Construct MLP class 
            mlp = MLP(input_features=feature_train_val.size()[1], hidden_units=hidden_units, output_units=1)

            # Construct loss and optimizer
            criterion = torch.nn.MSELoss(reduction="mean")
            optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate) # link to mlp parameters (lr should be 1e-2)
        else:
            pass

        # Train the model: Learning iteration
        mlp.train()
        
        # use trained model's weight of previous rolling sample as initial weight (continue training)
#         print("initial weight: ", mlp.state_dict()['hidden.weight'][0])

        total_step = len(train_val_loader)
        for epoch in range(num_epochs):
            for batch, (feature_train_val, target_train_val) in enumerate(train_val_loader):
#                 print(feature_train_val, target_train_val)
#                 print(feature_train_val.size(), target_train_val.size())
                # Forward pass
                target_pred = mlp(feature_train_val)
                # let y_pred be the same size as y
                target_pred = target_pred.squeeze(1)

                # Compute loss
                loss = criterion(target_pred, target_train_val) # link to mlp output
                # Zero gradients, perform backward pass, and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch+1) % 1000 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], : Loss {}'.format(epoch+1, num_epochs, batch+1, total_step, loss.item()))


        # Save the trained model
        PATH = '../../assets/trained_models/univariate/mlp/' + model_name + "_" + firm + "_" + "win" + str(num_epochs) + '.pth'
    #         torch.save(mlp.state_dict(), PATH)

        # use the existing trained model and continute training next window.
#         print("inherit weight: ", mlp.state_dict()['hidden.weight'][0])

        # predict y_hat (target_hat) <- 良くないかも、with torch_nograd() と model.eval()
        mlp.eval()
        with torch.no_grad():
            for feature_test, target_test in test_loader:
#                 print(feature_test, target_test)
#                 print(feature_test.size(), target_test.size())
                y_hat = mlp(feature_test)
#                 print(y_hat)
                y_hat_umlp.append(y_hat.squeeze().detach().numpy())

y_hat_umlp = pd.Series(y_hat_umlp)
y_hat_umlp.index = df.loc[pd.IndexSlice[firm, 2018:, :], :].index
y_hat_umlp.name = 'y_hat_umlp'
# y_hat_umlp.to_csv('../../assets/y_hats/univariate/y_hat_' + model_name + '.csv')

t2 = time.time()

elapsed_time = t2-t1
print(f"elapsedtime：{elapsed_time}")