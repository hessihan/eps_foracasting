# Forecasting TOYOTA's Earnings

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import warnings

warnings.simplefilter("ignore")

# reading csv
financial = pd.read_csv("data/input/FINFSTA_TOYOTA_199703_202004.csv", header=0)
nikkei_forecast = pd.read_csv("data/input/EARNING_TOYOTA_199703_202004.csv", header=0)
self_forecast = pd.read_csv("data/input/FINHISA_TOYOTA_199703_202003.csv", header=0)

# cleaning data
for df in [financial, nikkei_forecast, self_forecast]:
    df.drop([0, 1], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.replace("-", np.nan, inplace=True)

# selecting columns
eps_record = financial[[
    '決算期', '決算月数', '連結基準フラグ', '決算種別フラグ', '決算発表日', 
    '１株当たり利益［累計］', '１株当たり利益［３ヵ月］'
]]

# to datetime
for date_col in ["決算期", '決算発表日']:
    eps_record[date_col] = pd.to_datetime(eps_record[date_col])

# to float
eps_record = eps_record.astype({
    '決算月数': 'int8', '連結基準フラグ': 'int8', '決算種別フラグ': 'int8',
    '１株当たり利益［累計］': 'float64', '１株当たり利益［３ヵ月］': 'float64'
})
#print(eps_record.dtypes)

# plot record
fig = plt.figure(figsize=(16*2, 9))
ax = fig.add_subplot(111)
ax.scatter(eps_record["決算期"], eps_record["１株当たり利益［累計］"])
ax.plot(eps_record["決算期"], eps_record["１株当たり利益［累計］"])
ax.scatter(eps_record["決算期"], eps_record["１株当たり利益［３ヵ月］"])
ax.plot(eps_record["決算期"], eps_record["１株当たり利益［３ヵ月］"])
ax.set_xticks(eps_record["決算期"])
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.tick_params(axis="x", rotation=90)
plt.show()

# 2007/06 ~
start_period = eps_record[eps_record["決算期"] == "2007-06-01"].index.values[0]
ts = eps_record.loc[start_period:]
ts.reset_index(drop=True, inplace=True)

# fill nan for 3 months data
nan_index = ts["１株当たり利益［３ヵ月］"][ts["１株当たり利益［３ヵ月］"].isnull()].index.values[0]
ts["１株当たり利益［３ヵ月］"][ts["１株当たり利益［３ヵ月］"].isnull()] = ts.loc[nan_index]["１株当たり利益［累計］"] - ts.loc[nan_index - 1]["１株当たり利益［累計］"]

# y, x (lag 1 y) !!!!!!!!!!!!!!!!!!!!!!! lag 4 for quarterly
y = ts["１株当たり利益［３ヵ月］"].drop(0, axis=0)
y = y.reset_index(drop=True)
x = ts["１株当たり利益［３ヵ月］"].shift(1).drop(0, axis=0)
x = x.reset_index(drop=True)

# train (40) test (11) split
y_train = y[:40]
y_test = y[40:]
x_train = x[:40]
x_test = x[40:]

# ts forecast

# random walk
y_pred_rw = x_test

# SARIMA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! must run in R ? !!!!!!!!!!!!!!!!!!!!!
#https://blog.amedama.jp/entry/sm-decompose-series
#https://qiita.com/mshinoda88/items/749131478bfefc9bf365

import statsmodels.api as sm

# 定常性、季節性、トレンドのチェック
# plot ACF
sm.tsa.graphics.plot_acf(y)
plt.show()
# plot PACF
sm.tsa.graphics.plot_pacf(y)
plt.show()

# Brown & Rozeff (1, 0, 0) * (0, 1, 1)_4
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
y_pred_sarima_br = sarima_br.predict(y_test.index.values[0], y_test.index.values[-1])

# Griffin        (0, 1, 1) * (0, 1, 1)_4
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
y_pred_sarima_g = sarima_g.predict(y_test.index.values[0], y_test.index.values[-1])

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
y_pred_sarima_f = sarima_f.predict(y_test.index.values[0], y_test.index.values[-1])
# Box - Jenkins  (firm specific)


#ts.to_csv("data/output/sample_ts.csv")

# Neural Network
import module_nn




# plot each y_pred series
fig = plt.figure(figsize=(16*2, 9))
ax = fig.add_subplot(111)
ax.plot(eps_record["決算期"], eps_record["１株当たり利益［３ヵ月］"], marker="o", label="record")
    
ax.plot(eps_record["決算期"], [0] * len(eps_record["決算期"]), color="black")
ax.plot(ts["決算期"][41:], y_pred_rw, marker="o", label="random walk")
ax.plot(ts["決算期"][41:], y_pred_sarima_br, marker="o", label="SARIMA BR")
ax.plot(ts["決算期"][41:], y_pred_sarima_g, marker="o", label="SARIMA G")
ax.plot(ts["決算期"][41:], y_pred_sarima_f, marker="o", label="SARIMA F")

ax.set_xticks(eps_record["決算期"])
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.tick_params(axis="x", rotation=90)
ax.legend()
plt.show()