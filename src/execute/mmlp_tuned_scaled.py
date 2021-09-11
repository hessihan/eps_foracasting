# Run NN-MLP
# this is some sort of main execution file.

# import external libraries
import sys
import time
import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from tqdm import tqdm

# import internal modules
sys.path.insert(1, '../../src')
# from models.nn import MLP, TimeSeriesDataset
from models.nn import TimeSeriesDataset
from utils.data_editor import lag, train_test_split
from utils.accuracy import MAE, MAPE, MSE, Error, AbsoluteError, PercentageError, AbsolutePercentageError

# Define nn.Module subclass: MLP with optuna trial
# http://maruo51.com/2020/08/07/optuna_pytorch/
# https://qiita.com/Yushi1958/items/cd22ade638f7e292e520
# https://ichi.pro/optuna-o-shiyoshita-pytorch-haipa-parame-ta-no-chosei-4883072668892
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_ignite_simple.py
class MLP(torch.nn.Module):
    """
    a three-layer feedforward network with an identify
    transfer function in the output unit and logistic functions in the middle-layer
    units can approximate any continuous functions arbitrarily well, given sufficiently
    many middle-layer units.
    """
    def __init__(self, input_features, hidden_units, output_units=1):
        """
        Instantiate model layers.
        """
        super().__init__()
        # Fully connected layer
        # https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
        # hidden layer
        self.hidden = torch.nn.Linear(input_features, hidden_units, bias=True) 
        # output layer
        self.output = torch.nn.Linear(hidden_units, output_units, bias=True)
        
    def forward(self, x):
        """
        Forward pass: 
        """
        # input to hidden
        z = self.hidden(x)
        # logistic sigmoidal activation
        h = torch.sigmoid(z)
        # hidden to output
        out = self.output(h)
        # identify transfer function (do nothing)
        return out

def get_data_loaders(train_val_dataset, train_window_size):
    # 分け方は一応 full-psuedo で val 12, train 24でsplit。(val に rolling windowはしない)
    # batch はとりあえずなしで。
    train_dataset = torch.utils.data.dataset.Subset(train_val_dataset, list(range(0, train_window_size)))
    val_dataset = torch.utils.data.dataset.Subset(train_val_dataset, list(range(train_window_size, len(train_val_dataset))))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    return train_loader, val_loader

def objective(trial):
    # hyper paramater tuning space
    hidden_units = trial.suggest_int("hidden_units", 1e+0, 1e+3, step=1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-0, log=True)
    
    # instanciate model
    mlp = MLP(input_features=32, hidden_units=hidden_units, output_units=1)

    # Construct loss and optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    
    # get dataloader for this firm rolling sample
    train_loader, val_loader = get_data_loaders(train_val_dataset, TRAIN_WINDOW_SIZE)
    
    # -------------
    # Training Loop
    # -------------
    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        mlp.train()
        for batch, (feature_train, target_train) in enumerate(train_loader):
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
            if (epoch+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], : Loss {}'.format(epoch+1, NUM_EPOCHS, batch+1, total_step, loss.item()))

        # ---------------
        # Validation Loop (all batch)
        # ---------------
        mlp.eval()
        with torch.no_grad():
            for feature_val, target_val in val_loader:
                y_hat_val = mlp(feature_val)
        
        # return validation error
        # * Sum of Absolute Error
#         val_error = AbsoluteError(target_val.numpy(), y_hat_val.squeeze().detach().numpy()).sum()
        # * Sum of Squared Error
#         val_error = np.square(AbsoluteError(target_val.numpy(), y_hat_val.squeeze().detach().numpy())).sum()
        # * Mean Absolute Error
#         val_error = MAPE(target_val.numpy(), y_hat_val.squeeze().detach().numpy())
        # * Mean Squared Error
        val_error = MSE(target_val.numpy(), y_hat_val.squeeze().detach().numpy())
        
        trial.report(val_error, epoch)
        
#         Handle pruning based on the intermediate value. 
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_error
    
if __name__ == "__main__":
    print("mmlp start")
    # Set Seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Config
    NUM_EPOCHS = 10000
    PATIENCE = 5
    SCALE_X = True
    
    TRAIN_VAL_WINDOW_SIZE = 36
    TRAIN_WINDOW_SIZE = 30
    VAL_WINDOW_SIZE = TRAIN_VAL_WINDOW_SIZE - TRAIN_WINDOW_SIZE
    TEST_WINDOW_SIZE = 1
    BATCH_SIZE = None
    NUM_TRIALS = 100
    
    MODEL_NAME = "mmlp_tuned_scaled"
    PLOT_OPTUNA = False
    VERBOSE = False
    # ------------
    # Prepare Data
    # ------------
    # read processed data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    # small firm data setting for debugging
    df = df.loc[pd.IndexSlice[df.index.get_level_values(0).unique()[0], :, :], :]
#     df = df.loc[pd.IndexSlice[df.index.get_level_values(0).unique(), :, :], :]
    
    # empty list for agregated dataframes
    y_hat_mmlp_list = []
    log_list = []
    
    t1 = time.time() 

    # ------------
    # Loop Firm
    # ------------
    firm_list = df.index.get_level_values(0).unique()
    for firm in tqdm(firm_list):
        y_hat_mmlp = []

        # y : "EPS"
        y = df.loc[pd.IndexSlice[firm, :, :], "EPS"]

        # x, exogenous regressors : 'INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF'
        x = df.loc[pd.IndexSlice[firm, :, :], ['INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF']]

        # Unlike statsmodel SARIMA package, NN needs to prepare lagged inputs manually if needed.
        # y_lag and x_lag (lag 4 for now)
        num_lag = 4
        y_lag = lag(y, num_lag, drop_nan=False, reset_index=False)
        x_lag = lag(x, num_lag, drop_nan=False, reset_index=False)

        # Redefine data name as target (y) and feature (y_lag) (explanatory variable, predictor)
        target = y
        feature = pd.concat([y_lag, x_lag], axis=1)

        # save simple test data series
        rolling_sample_size = 12
        _, target_test_all = train_test_split(target, test_size=rolling_sample_size)
        _, feature_test_all = train_test_split(feature, test_size=rolling_sample_size)

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
        
        # prepare rolling sample dataset and loader
        train_val_all_dataset = TimeSeriesDataset(feature_all, target_all, train_window=TRAIN_VAL_WINDOW_SIZE)

        train_val_rolling_loader = torch.utils.data.DataLoader(train_val_all_dataset, batch_size=1, shuffle=False)

        # -------------------
        # Loop Rolling sample
        # -------------------
        for rolling_sample, (feature_train_val, target_train_val) in enumerate(train_val_rolling_loader):
#         for rolling_sample, (feature_train_val, target_train_val) in enumerate(tqdm(train_val_rolling_loader, leave=False)):
            if VERBOSE:
                print("")
                print("///////////////////////////////////////////////////")
                print("Firm: ", firm, "- Rolling Window: ", rolling_sample)
                print("///////////////////////////////////////////////////")
            #===============
            # Data
            #===============
            feature_train_val = feature_train_val[0] # extract single rolling sample batch
            target_train_val = target_train_val[0] # extract single rolling sample batch
            feature_test = feature_test_all[rolling_sample].reshape(1, -1)
            target_test = target_test_all[rolling_sample].reshape(-1)
            
            # Standard Scaling for training and validation
            if SCALE_X:
                # calculate mean and var based on only train (val excluded)
                feature_train_scaler = StandardScaler().fit(feature_train_val[:TRAIN_WINDOW_SIZE])
                # overwrite feature_train_valid memory
                feature_train_val[:TRAIN_WINDOW_SIZE] = torch.tensor(feature_train_scaler.transform(feature_train_val[:TRAIN_WINDOW_SIZE]), dtype=dtype)
                feature_train_val[TRAIN_WINDOW_SIZE:] = torch.tensor(feature_train_scaler.transform(feature_train_val[TRAIN_WINDOW_SIZE:]), dtype=dtype)
                feature_test = torch.tensor(feature_train_scaler.transform(feature_test), dtype=dtype)
                

            # all "batch" dataset (unit window)
            train_val_dataset = TimeSeriesDataset(feature_train_val, target_train_val, train_window=None)
            train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=len(train_val_dataset), shuffle=False)
            test_dataset = TimeSeriesDataset(feature_test, target_test, train_window=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            
            #=======================
            # Train fit and val eval
            #=======================
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
            study.optimize(objective, n_trials=NUM_TRIALS, timeout=600)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            if VERBOSE:
                print("Optuna Summary")
                print("Study statistics: ")
                print("  Number of finished trials: ", len(study.trials))
                print("  Number of pruned trials: ", len(pruned_trials))
                print("  Number of complete trials: ", len(complete_trials))

                print("Best trial:")
                print("  Value: ", study.best_trial.value)

                print("  Params: ")
                for key, value in study.best_trial.params.items():
                    print("    {}: {}".format(key, value))
                
            
            
            if PLOT_OPTUNA:
                print("Plotting Optimization Process for {} Rolling-Sample {}: ".format(firm, rolling_sample))
                fig = optuna.visualization.plot_optimization_history(study)
                fig.show()
                fig = optuna.visualization.plot_contour(study, params=["hidden_units", "learning_rate"])
                fig.show()
                fig = optuna.visualization.plot_slice(study, params=["hidden_units", "learning_rate"])
                fig.show()
                fig = optuna.visualization.plot_edf(study)
                fig.show()
                fig = optuna.visualization.plot_param_importances(study)
                fig.show()
            
            # save trials as dataframe
            study.trials_dataframe().to_csv("../../assets/trained_models/multivariate/mlp_tuned_scaled/optuna_trials/optuna_trials_" + firm + "_" + str(rolling_sample) + ".csv")

            #====================
            # Final Train_Val Fit
            #====================
            # best tuned hyparams
            best_hidden_units = study.best_trial.params["hidden_units"]
            best_learning_rate = study.best_trial.params["learning_rate"]
            
            # Instanciate model
            mlp = MLP(input_features=feature_train_val.size()[1], hidden_units=best_hidden_units, output_units=1)
            
            # Construct loss and optimizer
            criterion = torch.nn.MSELoss(reduction="mean")
            optimizer = torch.optim.Adam(mlp.parameters(), lr=best_learning_rate)
            
            # learning iteration
            total_step = len(train_val_loader)
            
            # |||||||||||||| Early stopping ||||||||||||||
            # initial values for early stopping
            patience = PATIENCE
            no_improve_epoch_count = 0
            min_loss = np.Inf
            early_stop = False
            # ||||||||||||||||||||||||||||||||||||||||||||
            
            # trained model save path
            PATH = "../../assets/trained_models/multivariate/mlp_tuned_scaled/" + MODEL_NAME + "_" + firm + "_" + str(rolling_sample) + '.pth'
            
            for epoch in range(NUM_EPOCHS):
                mlp.train()
                
                for batch, (feature_train_val, target_train_val) in enumerate(train_val_loader):
#                     print(feature_train_val, target_train_val)
#                     print(feature_train_val.size(), target_train_val.size())
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
                    if VERBOSE:
                        if (epoch+1) % (NUM_EPOCHS/10) == 0:
                            print('Epoch [{}/{}], Step [{}/{}], : Loss {}'.format(epoch+1, NUM_EPOCHS, batch+1, total_step, loss.item()))
            
                    # |||||||||||||| Early stopping ||||||||||||||
                    # Early stopping (monitor train_val_loss)
                    if loss < min_loss:
                        # if current loss is at a minimum
                        no_improve_epoch_count = 0
                        min_loss = loss
                        # Save and overwrite the (so far) best final trained model
                        torch.save(mlp.state_dict(), PATH)
                    else:
                        # if current epoch not improved
                        no_improve_epoch_count += 1
                        
                    if (epoch > patience) & (no_improve_epoch_count == patience):
                        early_stop = True
                        if VERBOSE:
                            print("* Early Stopping Executed at Epoch [{}/{}]".format(epoch+1, NUM_EPOCHS))
                            print("| no_improve_epoch_count", no_improve_epoch_count)
                            print("| min_loss", min_loss)
                            print("| early_stop", early_stop)
                        
                        # break current epoch step
                        break
                    
                if early_stop:
                    # break entire epoch loop
                    if VERBOSE:
                        print("Break epoch loop")
                    break
                    # ||||||||||||||||||||||||||||||||||||||||||||
            
            # load the best final trained model
            best_model = MLP(input_features=feature_train_val.size()[1], hidden_units=best_hidden_units, output_units=1)
            best_model.load_state_dict(torch.load(PATH))

            #======
            # Test
            #======
            best_model.eval()
            with torch.no_grad():
                for feature_test, target_test in test_loader:
    #                 print(feature_test, target_test)
    #                 print(feature_test.size(), target_test.size())
                    y_hat = best_model(feature_test)
                    y_hat_mmlp.append(y_hat.squeeze().detach().numpy())
#                     y_test_list.append(target_test)
                    if VERBOSE:
                        print("y_true: {}, y_hat: {}, diff: {}".format(target_test.squeeze().detach().numpy(), y_hat.squeeze().detach().numpy(), target_test.squeeze().detach().numpy() - y_hat.squeeze().detach().numpy()))
                        print("AbsolutePercentageError: ", AbsolutePercentageError(target_test.squeeze().detach().numpy(), y_hat.squeeze().detach().numpy()))
            
            # save log
            log_list.append([
                rolling_sample, 
                len(pruned_trials), 
                len(complete_trials), 
                study.best_trial.value,
                study.best_trial.params.items(),
                epoch+1,
                min_loss,
                target_test.squeeze().detach().numpy(),
                y_hat.squeeze().detach().numpy(),
            ])
            
        # print APEs and MAPE for current firm
        if VERBOSE:
            print("")
            print(firm, "accuray")
            print("AbsolutePercentageError: ", AbsolutePercentageError(target_test_all.detach().numpy(), np.array(y_hat_mmlp)))
            print("MAPE: ", MAPE(target_test_all.detach().numpy(), np.array(y_hat_mmlp)))
        
        # aggregate y_hat_mmlp for all firm y_hat_mmlp_list
        y_hat_mmlp_list.extend(y_hat_mmlp)
        
    # save as dataframe
#     y_test = df.loc[pd.IndexSlice[:, 2018:, :], "EPS"]
#     y_test.to_csv('../../assets/y_hats/multivariate/y_test_tuned.csv')
    
    y_hat_mmlp_list = pd.Series(y_hat_mmlp_list)
    y_hat_mmlp_list.index = df.loc[pd.IndexSlice[:, 2018:, :], :].index
    y_hat_mmlp_list.name = 'y_hat_mmlp'
    y_hat_mmlp_list.to_csv('../../assets/y_hats/multivariate/y_hat_' + MODEL_NAME + str(datetime.date.today()) + '.csv')
    
    log = pd.DataFrame(log_list)
    log.index = df.loc[pd.IndexSlice[:, 2018:, :], :].index
    log.columns = ["rolling_sample", "pruned_trials", "complete_trials", "best_trial_val_error", "best_trial_params", "final_train_epochs", "final_train_min_loss", "y_test", "y_hat"]
    log.to_csv("../../assets/y_hats/multivariate/log_tuned_scaled" + str(datetime.date.today()) + ".csv")
    
    t2 = time.time()

    elapsed_time = t2-t1
    print(f"elapsedtime：{elapsed_time}")