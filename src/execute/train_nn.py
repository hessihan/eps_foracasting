# this is some sort of main execution file.
# 

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
    from utils.data_editor import train_test_split, lag
    
    # Prepare Data
    
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
    
    # time series train test split (4/5) : (1/5), yearly bases
    y_train, y_test = train_test_split(y, ratio=(4,1))
    x_train, x_test = train_test_split(x, ratio=(4,1))
    
    # save y_test as csv for calculating accuracy indicators
    # y_test.name = "y_test"
    # y_test.to_csv('../../assets/y_hats/y_test.csv')
    
    # Unlike SARIMA package, NN needs to prepare lagged inputs if needed.
    # y_lag and x_lag (lag 4 for now)
    num_lag = 4
    y_train_lag = lag(y_train, num_lag, drop_nan=False, reset_index=False)
    x_train_lag = lag(x_train, num_lag, drop_nan=False, reset_index=False)

    # Define overall target (y_train) and features (y_train_lag and x_train_lag)
    target = y_train
    feature = pd.concat([y_train_lag, x_train_lag], axis=1)
    feature = feature.dropna(axis=0)
    target = target[feature.index]
    
    date = df["決算期"][target.index] # for plotting !!!! <-- 改善の余地あり, targetはtensorになってindexがなくなるから
    
    # setting torch
    dtype = torch.float # double float problem in layer 
    device = torch.device("cpu")
    
    # Make data to torch.tensor
    target = torch.tensor(target.values, dtype=dtype)
    feature = torch.tensor(feature.values, dtype=dtype)
    
    # Fit NN
    
    # MLP
    
    # Construct MLP class
    mlp = MLP(input_features=feature.size()[1], hidden_units=100, output_units=1)
    print(mlp)
    
    # Access to mlp weights
    print(list(mlp.parameters())) # iteretor, just for printing
    print(mlp.hidden.weight.size(), mlp.hidden.bias.size()) # editable ?
    print(mlp.output.weight.size(), mlp.output.bias.size())

    # Construct loss and optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2) # link to mlp parameters (lr should be 1e-2)

    # Train the model: Learning iteration
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(date, target.detach().numpy(), label="true")
    
    torch.manual_seed(0)
    num_iteration = 20000
    for step in range(num_iteration):
        # Forward pass
        target_pred = mlp(feature)
        # let y_pred be the same size as y
        target_pred = target_pred.squeeze(1)

        # Compute loss
        loss = criterion(target_pred, target) # link to mlp output
        if step % 1000 == 999:
            print(f"step {step}: loss {loss.item()}")
            ax.plot(date, target_pred.detach().numpy(), label="step: " + str(step))

        # Zero gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ax.legend()
    plt.show()
    
    # predict y_hat
    y_hat_mlp_mv = mlp(x_test_tensor).squeeze().detach().numpy()
    y_hat_mlp_mv
    
    # Save the trained model
    PATH = './lstm.pth'
    torch.save(model.state_dict(), PATH)