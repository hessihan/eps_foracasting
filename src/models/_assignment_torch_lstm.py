# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# https://dajiro.com/entry/2020/05/06/183255

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(1)

## Data Preprocessing

# データの読み込み
eps = pd.read_csv("annualEPS.csv", index_col=1)
eps = eps.drop('Unnamed: 0', axis=1)
eps = eps.T # 転置して時間をaxis=1に
eps = eps.iloc[::-1] # 時間を上から順に
eps.index = np.linspace(-29, 0, 30).astype(int)

# データのプロット
fig = plt.figure()
ax = fig.add_subplot()
plt.title('Annual EPS: CANON')
plt.ylabel('EPS (JPY)')
plt.xlabel('year')
plt.plot(eps['CANON INCORPORATED'])
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
plt.title('Annual EPS: All')
plt.ylabel('EPS (JPY)')
plt.xlabel('year')
plt.plot(eps)
plt.show()

all_data = eps['CANON INCORPORATED']

# 標準化(正規分布・minmax)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
all_normalized = scaler.fit_transform(all_data.values.reshape(-1, 1)) # expect 2D array

# Torch.Tensorに変換
all_normalized = torch.Tensor(all_normalized).view(-1)

# tuple で inout sequence 作成 (training window 指定、サンプルサイズは len(input_data)-tw に減る)
def create_inout_sequence(input_data, tw):
    inout_seq = []
    for i in range(len(input_data)-tw):
        train_seq = input_data[i: i+tw]
        train_label = input_data[i+tw: i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_window = 5
all_inout_seq = create_inout_sequence(all_normalized, train_window)

# 訓練データとテストデータに分割
# -29 ~ -5: train, -4 ~ 0: test
train_inout_seq = all_inout_seq[: -5]
test_inout_seq = all_inout_seq[-5: ]

## Creating LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        # LSTMは3つのインプットがある[previous hidden state, previous cell state, current input]
        # self.hidden_cellで[previous hidden state, previous cell state]を記憶しておく?
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size) # input_size: 入力次元, hidden_size: 隠れ層の次元
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                           torch.zeros(1, 1, self.hidden_layer_size))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
# make an LSTM() class object , define a loss function and optimizer
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # ここではAdam

## Training the Model
epochs = 1000 # 変えてもよい

for i in range(epochs):
    for seq, labels in train_inout_seq:
        # いろいろ0に初期化
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        
        y_pred = model(seq)
        
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# Save the trained model
PATH = './lstm.pth'
torch.save(model.state_dict(), PATH)

# Load the trained model
model = LSTM()
model.load_state_dict(torch.load(PATH))

## Making Predictions

# testをXとtに分割
test_X = []
test_label = []
for x, t in test_inout_seq:
    test_X.append(x)
    test_label.append(t)

# prediction loop
model.eval()

pred = []
for i in test_X:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        pred.append(model(i).item())

# convert normalized values to actual values
actual_pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1))

# 二乗平均平方根誤差 (RMSE: Root Mean Squared Error)で評価
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(eps['CANON INCORPORATED'][-5:].values, actual_pred.reshape(len(actual_pred)))
rmse = np.sqrt(mse)
print(rmse)

# plot
fig = plt.figure()
ax = fig.add_subplot()

plt.plot(eps['CANON INCORPORATED'], label='actual') # all data
x_axis = np.linspace(-29, 0, 30).astype(int)
plt.plot(x_axis[-5:], actual_pred, label='prediction')

plt.title('Annual EPS: CANON')
plt.ylabel('EPS (JPY)')
plt.xlabel('year')
plt.legend()

plt.show()