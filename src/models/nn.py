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
        self.hidden = torch.nn.Linear(input_features, hidden_units) 
        # hidden to Output layer, 1 units
        self.output = torch.nn.Linear(hidden_units, output_units)
        
    def forward(self, x):
        """
        Forward pass: 
        """
        # input to hidden
        x = self.hidden(x)
        # logistic sigmoidal activation
        x = torch.sigmoid(x)
        # hidden to output
        x = self.output(x)
        # identify transfer function (do nothing)
        return x

# Define nn.Module subclass: LSTM
class LSTM(torch.nn.Module):
    """
    long short-term memory network
    """
    def __init__(self, input_features=4, hidden_units=100, output_units=1):
        "Instantiate model layers."
        super(LSTM, self).__init__()

# save model
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
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