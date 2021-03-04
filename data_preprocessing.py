import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import warnings

warnings.simplefilter("ignore")

# reading input csv
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

ax.plot(eps_record["決算期"], eps_record["１株当たり利益［累計］"], marker="o", label="annual EPS")
ax.plot(eps_record["決算期"], eps_record["１株当たり利益［３ヵ月］"], marker="o", label="quaterly EPS")

ax.set_xticks(eps_record["決算期"])
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.tick_params(axis="x", rotation=90)
ax.legend()
plt.show()

# extract data 2007/06 ~
start_period = eps_record[eps_record["決算期"] == "2007-06-01"].index.values[0]
ts = eps_record.loc[start_period:]
ts.reset_index(drop=True, inplace=True)

# fill nan for 3 months data
nan_index = ts["１株当たり利益［３ヵ月］"][ts["１株当たり利益［３ヵ月］"].isnull()].index.values[0]
ts["１株当たり利益［３ヵ月］"][ts["１株当たり利益［３ヵ月］"].isnull()] = ts.loc[nan_index]["１株当たり利益［累計］"] - ts.loc[nan_index - 1]["１株当たり利益［累計］"]

# Export cleaned data
ts.to_csv("data/output/sample_ts.csv")
