# Run NN-LSTM
# this is some sort of main execution file.
# https://github.com/yunjey/pytorch-tutorial

if __name__ == "__main__":
    
    ## Libraries
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import torch
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.nn import LSTM
    from utils.data_editor import lag, train_test_split, TimeseriesDataset
    
    # set seeds for reproductibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    ## Prepare Data
    
    # read processed data
    df = pd.read_csv("../../dataset/processed/dataset.csv")

    # save column names
    earning_v = df.columns[4: 10].values
    account_v_bs = df.columns[11:].values
    account_v_pl = df.columns[10:11].values

    # y: "１株当たり利益［３ヵ月］"
    y = df[earning_v[-1]]
    
    # x: ['棚卸資産', '資本的支出', '期末従業員数', '受取手形・売掛金／売掛金及びその他の短期債権', 
    #    '販売費及び一般管理費']
    x = df[np.append(account_v_bs, account_v_pl)]
    
    # Unlike MLP, LSTM needs to prepare lagged inputs with seq_len matrix.
    # feature must be only lag1 (y||x)
    num_lag = 1
    y_lag = lag(y, num_lag, drop_nan=False, reset_index=False)
    x_lag = lag(x, num_lag, drop_nan=False, reset_index=False)

    # Redefine data name as target (y) and feature (y_lag and x_lag)
    target = y
    feature = pd.concat([y_lag, x_lag], axis=1)
    
    # time series train test split (4/5) : (1/5), yearly bases
    # DataLoader使うからTrainとtestぶつ切りにしたらtest時サンプル減る--> testに"seq_len - 1"だけ加える
    target_train, target_test = train_test_split(target, ratio=(4,1))
    feature_train, feature_test = train_test_split(feature, ratio=(4,1))

    train_date = df["決算期"][target_train.index] # for plotting <-- 改善の余地あり, targetはtensorになってindexがなくなるから
    test_date = df["決算期"][target_test.index] # for y_hat index <-- 改善の余地あり, targetはtensorになってindexがなくなるから    

    # !!!!! HYPARAM set sequence length !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    seq_len = 4 # training_window for one step prediction ## HYPARAM
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # add time length of "seq_len - 1" to test (for seq_len DataLoader)
    feature_test = pd.concat([feature_train[-(seq_len-1) :], feature_test], axis=0)
    target_test = pd.concat([target_train[-(seq_len-1) :], target_test], axis=0)
    
    # drop nan in train data head caused by lag() (only 1 lag)
    feature_train = feature_train.dropna(axis=0)
    target_train = target_train[feature_train.index]
    
    # setting torch
    dtype = torch.float # double float problem in layer 
    device = torch.device("cpu")
    
    # Make data to torch.tensor
    target_train = torch.tensor(target_train.values, dtype=dtype)
    feature_train = torch.tensor(feature_train.values, dtype=dtype)
    target_test = torch.tensor(target_test.values, dtype=dtype)
    feature_test = torch.tensor(feature_test.values, dtype=dtype)

    # Data Loader
    train_dataset = TimeseriesDataset(feature_train, target_train, seq_len=seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = False)
    
    test_dataset = TimeseriesDataset(feature_test, target_test, seq_len=seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
    
    ## Train LSTM
    
    # !!!!! HYPARAM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    hidden_dim = 500
    learning_rate = 1e-3
    num_epochs = 1000
#     num_layers = 1 # rnn hidden layer数
#     batch_size = 1 # ミニバッチしたくないからバッチサイズ=1でいいってことだよね?(塊でやらない)
    # Optimizer
    model_name = 'lstm_mv_hid' + str(hidden_dim) + "_lr" + str(learning_rate) + "_epoch" + str(num_epochs)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # Instantiate LSTM class
        
    lstm = LSTM(input_dim=feature_train.size()[1], hidden_dim=hidden_dim, output_dim=1)
#     print(lstm)

    # Construct loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model: Learning loop
    print("Train Model: " + model_name)
#     total_step = len(feature_train)
    for epoch in range(num_epochs):
        # LSTMはtを1つずつ進めて学習?
        for t, (feature_t, target_t) in enumerate(train_loader):
            # feature_t; x_t (seq_len x D_x)
            # torch.nn.LSTM() に渡すために変形 input: (seq_len, batch, input_size)
#             feature_t = feature_t.view(1, seq_len, input_dim) # (1, 1, 24)ではなく(batch, seq_len, input_size) = (1, 4, 6): batch_size=True

            # Forward pass
            target_t_pred = lstm(feature_t)
            loss = criterion(target_t_pred.view(1), target_t) # size()をtorch.Size([1])にそろえる

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, num_epochs, t+1, total_step, loss.item()))
        if (epoch+1) % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))
    
    
    ## Test Model: predict y_hat
    lstm.eval()
    # predict
    y_hat_lstm = []
    with torch.no_grad():
        for feature_t, target_t in test_loader:
            y_hat = lstm(feature_t)
            y_hat_lstm.append(y_hat.item())

    # to DataFrame and save as csv
    y_hat_lstm = pd.Series(y_hat_lstm)
    y_hat_lstm.name = 'y_hat_' + model_name
    y_hat_lstm.index = test_date
    y_hat_lstm.to_csv('../../assets/y_hats/y_hat_' + model_name + '.csv')
    
    ## Save trained model
    torch.save(lstm.state_dict(), '../../assets/trained_models/'+ model_name +'.pth')
