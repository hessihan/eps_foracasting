# Neural Networks module
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://qiita.com/hoolly728/items/c398afd5a21669b8ce0f
# https://hilinker.hatenablog.com/entry/2018/06/23/204910
# https://dajiro.com/entry/2020/05/06/183255
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

# Pytorch Neural Networks Scripting

import torch
import math
import pandas as pd

# Time Series torch Dataset class for rolling window process preparation
# https://discuss.pytorch.org/t/dataloader-for-a-lstm-model-with-a-sliding-window/22235
# https://stackoverflow.com/questions/57893415/pytorch-dataloader-for-time-series-task
# for pandas dataframe version, check below
# https://stackoverflow.com/questions/53791838/elegant-way-to-generate-indexable-sliding-window-timeseries
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
        window : int
             window size for each rolling window

        Return
        ------
        len(target) - train_window == len(target_test) sets of feature_train and target_train tensors (sibgel batch axis = 0)
        overview:
            ((batch_size=1, train_window, num_features), (batch_size=1, train_window))
            ...
            ((batch_size=1, train_window, num_features), (batch_size=1, train_window)). 
            # len(target) - train_window == len(target_test)
        """
        self.target = target
        self.feature = feature
        self.train_window = train_window

    def __len__(self):
        # num of output rolling window sets
        return len(self.target) - self.train_window

    def __getitem__(self, index):
        train_feature_window = self.feature[index: index + self.train_window]
        train_target_window = self.target[index: index + self.train_window]
        return train_feature_window, train_target_window

# Define nn.Module subclass: MLP
class MLP(torch.nn.Module):
    """
    a three-layer feedforward network with an identify
    transfer function in the output unit and logistic functions in the middle-layer
    units can approximate any continuous functions arbitrarily well, given sufficiently
    many middle-layer units.
    """
    def __init__(self, input_features=None, hidden_units=100, output_units=1):
        """
        Instantiate model layers.
        """
        super().__init__()
        # Fully connected layer
        # https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
        # Inputs to hidden layer, affine operation: y = Wx + b
        # How many units should be in hidden layers? 100 units for now. 
        self.hidden = torch.nn.Linear(input_features, hidden_units, bias=True) 
        # hidden to Output layer, 1 units
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

class LSTM(torch.nn.Module):
    """
    long short-term memory network.
    Nso batches (batch_size=1), single hidden lstm layer.
        
    Parameters
    ----------
    input_size : int
        the number of features in the input layer
    hidden_size : int
        the number of units (neurons) in each hidden layer
        (single hidden lstm layer for now)
    output_size : int, Default: 1
        the number of dimension for the output

    !! value changes not recomemnded !!
    num_layers : int, Default: 1
        Number of recurrent layers
        (single lstm layer for now)
    batch_size : int, Default: 1
        size of batches
        (No batches for now; batch_size=1)
    
    Reference :
    https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python
    https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb
    https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py#L39-L58
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, batch_size=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        # the layers
        # first layer (hidden lstm layer)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.hidden_lstm = torch.nn.LSTM(input_size=input_dim, 
                                         hidden_size=hidden_dim, 
                                         # num_layers=num_layers, 
                                         bias=True,
                                         batch_first=True)
        # second layer (linear output layer)
        self.output = torch.nn.Linear(hidden_dim, 
                                      output_dim, 
                                      bias=True)

    def forward(self, x):
        """
        Set initial hidden and cell states as zeros (stateless LSTM)
        and forward propagate LSTM.

        math::
            \begin{array}{ll} \\
                i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
                f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
                g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
                o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
                c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
                h_t = o_t \odot \tanh(c_t) \\
            \end{array}
        
        :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell state at time `t`.
        
        (input, (h_0, c_0)) are the inputs and
        (output, (h_n, c_n)) are the returns of lstm layer for t = n lstm cell.
        
        Parameters
        ----------
        x: inputs; shape (seq_len, batch_size=1, input_dim)
            tensor containing the features of the input sequence. 
            must include batch_size(=1) dimension as dim=1
        
        Returns
        -------
        h: 
        """
        # reset_hidden_state (stateless LSTM)
        # hidden state (lstm層のアクティベーション) of shape (batch, num_layers, hidden_size): batch_first=True
        h_0 = torch.zeros(self.batch_size, 1 * self.num_layers, self.hidden_dim)
        # cell state (メモリセル) of shape (batch, num_layers, hidden_size): batch_first=True
        c_0 = torch.zeros(self.batch_size, 1 * self.num_layers, self.hidden_dim)
        
        # Forward pass
        # input to hidden LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_size): batch_first=True
        h_all_time, (h_latest, c_latest) = self.hidden_lstm(x, (h_0, c_0))
        
        # change shape of h_all_time from (batch, num_layers, hidden_size) to (batch*num_layers, hidden_size)
        h = h_latest.view(-1, self.hidden_dim)
        
        # hidden LSTM layer to output layer
        out = self.output(h)
        return out
    
# Debugging
if __name__ == "__main__":
    
    # Reading data, convert pandas.DataFrame to torch.tensor
    ts = pd.read_csv("data/cleaned/sample_ts.csv")

    # y, x (lag 4 for now)
    y = ts["１株当たり利益［３ヵ月］"].drop([0, 1, 2, 3], axis=0)
    y = y.reset_index(drop=True)

    x = pd.DataFrame([
        ts["１株当たり利益［３ヵ月］"].shift(1),
        ts["１株当たり利益［３ヵ月］"].shift(2),
        ts["１株当たり利益［３ヵ月］"].shift(3),
        ts["１株当たり利益［３ヵ月］"].shift(4),
    ]).T.drop([0, 1, 2, 3], axis=0)
    x = x.reset_index(drop=True)

    x_label = ts["決算期"].drop([0, 1, 2, 3], axis=0)
    x_label = x_label.reset_index(drop=True)

    # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/14
    dtype = torch.float # double float problem in layer 
    device = torch.device("cpu")

    y = torch.tensor(y.values, dtype=dtype)
    x = torch.tensor(x.values, dtype=dtype)

    ######################## checking layers behavior #####################################
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    m = torch.nn.Linear(4, 3, bias=True)
    print(m.weight, m.bias)
    output = m(x) # y = x(m1) @ w^T(2) + b
    print(output.size())
    #######################################################################################

    # Construct mlp
    mlp = MLP()
    print(mlp)
    # Access to mlp weights
    print(list(mlp.parameters())) # iteretor, just for printing
    print(mlp.hidden.weight.size(), mlp.hidden.bias.size()) # editable ?
    print(mlp.output.weight.size(), mlp.output.bias.size())

    # Construct loss and optimizer
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2) # link to mlp parameters (lr should be 1e-2)

    # Learning iteration
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(x_label, y.detach().numpy(), label="true")

    num_iteration = 20000
    for step in range(num_iteration):
        # Forward pass
        y_pred = mlp(x)
        # let y_pred be the same size as y
        y_pred = y_pred.squeeze(1)

        # Compute loss
        loss = criterion(y_pred, y) # link to mlp output
        if step % 1000 == 999:
            print(f"step {step}: loss {loss.item()}")
            ax.plot(x_label, y_pred.detach().numpy(), label="step: " + str(step))

        # Zero gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ax.legend()
    plt.show()