# Forecasting TOYOTA's Earnings

import os
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# run data_preprocessing.py file to generate data
os.system("python data_preprocessing.py")

# read cleaned data generated from data_preprocessing.py
ts = pd.read_csv("data/output/sample_ts.csv", index_col=0)

# y, x (lag 1 y) 
# !!!!!! lag 4 for quarterly ? or statsmodel have done well in sm.tsa.SARIMAX?
y = ts["１株当たり利益［３ヵ月］"].drop(0, axis=0)
y = y.reset_index(drop=True)
x = ts["１株当たり利益［３ヵ月］"].shift(1).drop(0, axis=0)
x = x.reset_index(drop=True)

# train (40) test (11) split
y_train = y[:40]
y_test = y[40:]
x_train = x[:40]
x_test = x[40:]

# Check stationality, seasonality, and trends.
# plot ACF
sm.tsa.graphics.plot_acf(y)
plt.show()
# plot PACF
sm.tsa.graphics.plot_pacf(y)
plt.show()

# Model Estimation

## random walk
y_hat_rw = x_test

# SARIMA

# Brown & Rozeff (1, 0, 0) * (0, 1, 1)_4

# model estimation
sarima_br = sm.tsa.SARIMAX(
    endog=y_train, 
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()
print(sarima_br.summary())

# residual ACF PACF
resid_br = sarima_br.resid
sm.graphics.tsa.plot_acf(resid_br)
sm.graphics.tsa.plot_pacf(resid_br)
plt.show()

# predict test period !!!!!!!!!!!!!!!!!!!!!! not moving or increasing window
y_hat_sarima_br = sarima_br.predict(y_test.index.values[0], y_test.index.values[-1])

# Griffin        (0, 1, 1) * (0, 1, 1)_4

# model estimation
sarima_g = sm.tsa.SARIMAX(
    endog=y_train, 
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()
print(sarima_g.summary())

# residual ACF PACF
resid_g = sarima_g.resid
sm.graphics.tsa.plot_acf(resid_g)
sm.graphics.tsa.plot_pacf(resid_g)
plt.show()

# predict test period !!!!!!!!!!!!!!!!!!!!!! not moving or increasing window
y_hat_sarima_g = sarima_g.predict(y_test.index.values[0], y_test.index.values[-1])

# Foster         (1, 0, 0) * (0, 1, 0)_4
sarima_f = sm.tsa.SARIMAX(
    endog=y_train, 
    order=(1, 0, 0),
    seasonal_order=(0, 1, 0, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()
print(sarima_f.summary())
# residual ACF PACF
resid_f = sarima_f.resid
sm.graphics.tsa.plot_acf(resid_f)
sm.graphics.tsa.plot_pacf(resid_f)
plt.show()
# predict test period !!!!!!!!!!!!!!!!!!!!!! not moving or increasing window
y_hat_sarima_f = sarima_f.predict(y_test.index.values[0], y_test.index.values[-1])

# Box - Jenkins  (firm specific)

# Neural Network
import module_nn


# Predition Performance

# MAPE
# MSE

# plot each y_hat series
fig = plt.figure(figsize=(16*2, 9))
ax = fig.add_subplot(111)
ax.plot(ts["決算期"], [0] * len(ts["決算期"]), color="black")

ax.plot(ts["決算期"], ts["１株当たり利益［３ヵ月］"], marker="o", label="record")
    

ax.plot(ts["決算期"][41:], y_hat_rw, marker="o", label="random walk")
ax.plot(ts["決算期"][41:], y_hat_sarima_br, marker="o", label="SARIMA BR")
ax.plot(ts["決算期"][41:], y_hat_sarima_g, marker="o", label="SARIMA G")
ax.plot(ts["決算期"][41:], y_hat_sarima_f, marker="o", label="SARIMA F")

ax.tick_params(axis="x", rotation=90)
ax.legend()
plt.show()