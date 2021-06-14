import torch

torch.manual_seed(0)

seq_len = 100 # レコード期間
input_size = 4 # 特徴量数
hidden_size = 10 # hidden unit数
num_layers = 1 # rnn hidden layer数
batch_size = seq_len # ミニバッチしたくないけど、lstm(x, (h0, c0))ではbatch_sizeを考慮した行列が必要なので、無理やりレコード期間と同じバッチサイズを指定。

lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# check initial weights
# the learnable input-hidden weights of the \text{k}^{th}k th layer (W_ii|W_if|W_ig|W_io), 
# of shape (4*hidden_size, input_size) for k = 0. 
# Otherwise, the shape is (4*hidden_size, num_directions * hidden_size)
print(lstm.weight_ih_l0)
print(lstm.weight_ih_l0.size())
# the learnable hidden-hidden weights of the \text{k}^{th}k th layer (W_hi|W_hf|W_hg|W_ho), 
# of shape (4*hidden_size, hidden_size)
print(lstm.weight_hh_l0)
print(lstm.weight_hh_l0.size())
# bias
print(lstm.bias_ih_l0)
print(lstm.bias_ih_l0.size())
print(lstm.bias_hh_l0)
print(lstm.bias_hh_l0.size())

# input of shape (seq_len, batch, input_size)
x = torch.randn(seq_len, batch_size, input_size)

# hidden state (lstm層のアクティベーション) of shape (num_layers, batch, hidden_size):
h0 = torch.randn(num_layers, x.size(1), hidden_size)

# cell state (メモリセル) of shape (num_layers, batch, hidden_size): 
c0 = torch.randn(num_layers, x.size(1), hidden_size)

# return of LSTM layer
output, (hn, cn) = lstm(x, (h0, c0))

# check size of returns
print(output.size())
print(hn.size())
print(cn.size())

















































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
            (single hidden layer for now)
        num_layers : int, Default: 1
            the number of hidden layers?
            (single hidden layer for now)
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
        """
        # reset_hidden_state (stateless LSTM)
        # セルの中身は毎回やり直し、hiddenの値だけを影響して受け継いでいく感じ
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        # Forward pass
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.output(h_out)
        return out