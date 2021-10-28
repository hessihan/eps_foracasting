import os
import numpy as np
import pandas as pd
import datetime

## CAPM
def capm(r_i):
    """
    Monthly CAPM
    """
    return None

## 3 Factor
# https://s-tkmt.hatenablog.com/entry/2021/02/11/214230

## Conditional 3 Factor

## get cost of capital

## RIM
def rim(bps, r_e, feps, TV="const"):
    """
    Residual Income Model.
    Giving current BPS_{t}, cost of capital, and future EPS_{t+1} and returns intrinsic value of a certain stock.

    Paramteters
    -----------
    TV:
        terminal value setting
        "const": assume RI_t is same for t --> inf
    """
    v = bps + ((feps - r_e * bps) / ((1 + r_e) * r_e)) * bps
    return v
## VPR


## Data

# working dir
wd = "/mnt/d/0ngoing/thesis/repo"

# risk free rate
rf = pd.read_csv(wd + "/data/raw/MFJ/jgbcm_all_utf8.csv")
rf = rf.replace("-", np.nan)
rf[rf.columns[1:]] = rf[rf.columns[1:]].astype(float)


def jp_calendar_to_west_calendar(x):
    # https://tokukita.jp/hayami/wareki-seireki.html
    d = list(map(int, x[1:].split(".")))
    if x[0] == "S":
        d[0] += 1925
    elif x[0] == "H":
        d[0] += 1988
    elif x[0] == "R":
        d[0] += 2018
    d = datetime.date(d[0], d[1], d[2])
    return d

rf["t"] = rf["t"].apply(jp_calendar_to_west_calendar)

# market monthly return
rm = pd.read_excel(wd + "/data/raw/FQ/MKTINDEX_monthly/MarketIndex_copied.xlsx")
rm = rm[["東証.2", "東証.3"]]
rm.columns = ["end_mon_date", "price"]
rm = rm.drop([0, 1, 2, 3, 4, 5])
rm["t"] = rm["end_mon_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d"))
pd.merge(rm, rf, on="t")

# read y_hats_all.csv
y_hats_all = pd.read_csv(wd + "/assets/y_hats/y_hats_all.csv", index_col=[0, 1, 2])

y_hat_men_i_tuned_simple = y_hats_all["y_hat_men_i_tuned_simple"]

rim(100, 0.1, 10)