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

# Define nn.Module subclass: LSTM
class LSTM(torch.nn.Module):
#     def __init__(self, input_features=4, hidden_units=100, output_units=1):
#         "Instantiate model layers."
#         super().__init__()

    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        # LSTMは3つのインプットがある[previous hidden state, previous cell state, current input]
        # self.hidden_cellで[previous hidden state, previous cell state]を記憶しておく
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size) # input_size: 入力次元, hidden_size: 隠れ層の次元
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                           torch.zeros(1, 1, self.hidden_layer_size))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
class LSTM(torch.nn.Module):
    """
    long short-term memory network
    
    Reference :
    https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python
    https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb
    https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py#L39-L58
    """
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super().__init__()
        """
        Instantiate model layers.
        
        Parameters
        ----------
        input_size : int
            the number of features in the input layer
        hidden_size : int
            the number of units (neurons) in each hidden layer
            (single hidden lstm layer for now)
        num_layers : int, Default: 1
            Number of recurrent layers
            (single lstm layer for now)
        output_size : int, Default: 1
            the number of dimension for the output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # the layers
        # first layer (lstm layer)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = torch.nn.LSTM(input_size=input_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_layers, 
                                  bias=True)
        # second layer (linear output layer)
        self.output = torch.nn.Linear(hidden_size, output_size, bias=True)

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
        (output, (h_n, c_n)) are the returns of lstm layer.
        
        Parameters
        ----------
        input of shape (seq_len, batch, input_size): 
            tensor containing the features of the input sequence. 
            !! ミニバッチを使わないとしても batch を指定したtensorが必要 !!
            そもそも、batch はコンピューターサイエンスではサンプルサイズとして捉えられるらしい。
        """
        # reset_hidden_state (stateless LSTM)
        # セルの中身は毎回やり直し、hiddenの値だけを影響して受け継いでいく感じ
        h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
        
        # Forward pass
        # input to hidden LSTM layer
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        # change shape of h_out from (num_layers, batch, hidden_size) to (num_layers*batch, hidden_size)
        h_out = h_out.view(-1, self.hidden_size)
        
        # hidden LSTM layer to 
        out = self.output(h_out)
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