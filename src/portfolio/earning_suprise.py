# Buy if y_hat higher than stock_price, sell if lower.

import os
import datetime
import numpy as np
import pandas as pd

wd = "/mnt/d/0ngoing/thesis/repo/"

def read_stock_price(dir_path):
    stock_price = pd.DataFrame()
    for file in os.listdir(dir_path):
        # Define path
        data_file_path = dir_path + file
        # read temporary data file
        d = pd.read_excel(data_file_path)
        d = d.drop([0, 1])
        d = d.reset_index(drop=True)
        d = d[["Unnamed: 0", "月間始値日", '月間始値', "月間終値日", '月間終値']]
        d.columns = ["name", "start_month_date", "start_month_price", "end_month_date", "end_month_price"]
        d["start_month_date"] = d["start_month_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d").date())
        d["end_month_date"] = d["end_month_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d").date())
        d["name"] = d["name"].astype("str")
        d["start_month_price"] = d["start_month_price"].astype("float")
        d["start_month_date"] = d["start_month_date"].astype("datetime64")        
        d["end_month_price"] = d["end_month_price"].astype("float")
        d["end_month_date"] = d["end_month_date"].astype("datetime64")        
        stock_price = pd.concat([stock_price, d], axis=0)
        print(file + " concated")
    # stock_price = stock_price.set_index(["Type", "会計年度", "四半期"])
    return stock_price

path = "/mnt/d/0ngoing/thesis/repo/data/raw/FQ/Stock_monthly/test_period/"
stock_price = read_stock_price(path)
stock_price = stock_price.set_index("name")

y_hats_all = pd.read_csv(wd + "/assets/y_hats/y_hats_all.csv", index_col=[0, 1, 2])
y_hat_ibes = pd.read_csv(wd + "/data/processed/y_hat_ibes.csv", index_col=[0, 1, 2])
y_hat_ibes.index.get_level_values(0).unique()

# target firm (ibes target 99 firms)
firms = y_hat_ibes.index.get_level_values(0).unique() & stock_price.index.unique()

stock_price = stock_price.loc[firms]
# stock_price.to_csv(wd + "/data/processed/stock_price.csv")

y_hats_all = y_hats_all.loc[firms]
