# Run NN-LSTM
# this is some sort of main execution file.
# https://github.com/yunjey/pytorch-tutorial

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
    df = pd.read_csv("../../data/processed/dataset.csv")

    # save column names
    earning_v = df.columns[4: 10].values
    account_v_bs = df.columns[11:].values
    account_v_pl = df.columns[10:11].values

    # y: "１株当たり利益［３ヵ月］"
    y = df[earning_v[-1]]
    
    # x: ['棚卸資産', '資本的支出', '期末従業員数', '受取手形・売掛金／売掛金及びその他の短期債権', 
    #     '販売費及び一般管理費']
    x = df[np.append(account_v_bs, account_v_pl)]
    
    # Unlike statsmodel SARIMA package, NN needs to prepare lagged inputs manually if needed.
    # y_lag and x_lag (lag 4 for now)
    num_lag = 4
    y_lag = lag(y, num_lag, drop_nan=False, reset_index=False)
    x_lag = lag(x, num_lag, drop_nan=False, reset_index=False)

    # Redefine data name as target (y) and feature (y_lag and x_lag)
    target = y
    feature = pd.concat([y_lag, x_lag], axis=1)
    
    # time series train test split (4/5) : (1/5), yearly bases
    target_train, target_test = train_test_split(target, ratio=(4,1))
    feature_train, feature_test = train_test_split(feature, ratio=(4,1))
    
    # drop nan caused by lag()
    feature_train = feature_train.dropna(axis=0)
    target_train = target_train[feature_train.index]
    
    train_date = df["決算期"][target_train.index] # for plotting !!!! <-- 改善の余地あり, targetはtensorになってindexがなくなるから
    
    # setting torch
    dtype = torch.float # double float problem in layer 
    device = torch.device("cpu")
    
    # Make data to torch.tensor
    target_train = torch.tensor(target_train.values, dtype=dtype)
    feature_train = torch.tensor(feature_train.values, dtype=dtype)
    target_test = torch.tensor(target_test.values, dtype=dtype)
    feature_test = torch.tensor(feature_test.values, dtype=dtype)

    # Train MLP
    
    ### ! Hyper-Parameter ! ##########################################################
    hidden_units = 1000
    learning_rate = 1e-3
    num_epoch = 20000
    # Optimizer
    
    
    model_name = 'mlp_mv_hid1k_lr1en3_iter20k'
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

    # Train the model: Learning iteration
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(train_date, target_train.detach().numpy(), label="true")
    
    for step in range(num_iteration):
        # Forward pass
        target_pred = mlp(feature_train)
        # let y_pred be the same size as y
        target_pred = target_pred.squeeze(1)

        # Compute loss
        loss = criterion(target_pred, target_train) # link to mlp output
        if step % 1000 == 999:
            print(f"step {step}: loss {loss.item()}")
            ax.plot(train_date, target_pred.detach().numpy(), label="step: " + str(step))

        # Zero gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ax.legend()
    fig.savefig("../../assets/y_hats/train_process_" + model_name + ".png", format="png", dpi=300)
    plt.show()
    
    
    # predict y_hat (target_hat)
    y_hat_mlp_mv = mlp(feature_test).squeeze().detach().numpy()
    y_hat_mlp_mv = pd.Series(y_hat_mlp_mv)
    y_hat_mlp_mv.name = 'y_hat_mlp_mv'
    y_hat_mlp_mv.to_csv('../../assets/y_hats/y_hat_' + model_name + '.csv')
    
    # Save the pre_trained model
    PATH = '../../assets/trained_models/' + model_name + '.pth'
    torch.save(mlp.state_dict(), PATH)