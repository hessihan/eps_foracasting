# Buy if y_hat higher than stock_price, sell if lower.

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        d.columns = ["企業名", "start_month_date", "start_month_price", "end_month_date", "end_month_price"]
        d["start_month_date"] = d["start_month_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d").date())
        d["end_month_date"] = d["end_month_date"].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d").date())
        d["企業名"] = d["企業名"].astype("str")
        d["start_month_price"] = d["start_month_price"].astype("float")
        d["start_month_date"] = d["start_month_date"].astype("datetime64")        
        d["end_month_price"] = d["end_month_price"].astype("float")
        d["end_month_date"] = d["end_month_date"].astype("datetime64")        
        stock_price = pd.concat([stock_price, d], axis=0)
        print(file + " concated")
    # stock_price = stock_price.set_index(["Type", "会計年度", "四半期"])
    return stock_price

# path = "/mnt/d/0ngoing/thesis/repo/data/raw/FQ/Stock_monthly/test_period/"
# stock_price = read_stock_price(path)
# stock_price = stock_price.set_index("企業名")
# stock_price.to_csv(wd + "/data/processed/stock_price.csv")
stock_price = pd.read_csv(wd + "/data/processed/stock_price.csv")
stock_price["end_month_date"] = pd.to_datetime(stock_price["end_month_date"])
# https://www.advan.co.jp/pdf/company/ir/press-2021.04.26.pdf
stock_price["企業名"][stock_price["企業名"] == "アドヴァン"] = "アドヴァングループ"

y_hats_all = pd.read_csv(wd + "/assets/y_hats/y_hats_all.csv", index_col=[0, 1, 2])
y_hat_ibes = pd.read_csv(wd + "/data/processed/y_hat_ibes.csv", index_col=[0, 1, 2])

all_firm = y_hats_all.index.get_level_values(0).unique()
ibes_firm = y_hat_ibes.index.get_level_values(0).unique()

# Check release date
tidy_tse1 = pd.read_csv("../../data/processed/tidy_tse1.csv", index_col=[0, 1, 2])
release_date = tidy_tse1["決算発表日"]
release_date = pd.to_datetime(release_date)

release_date_test_period = release_date.loc[pd.IndexSlice[all_firm, [2017, 2018, 2019, 2020], :]]
release_date_test_period = release_date_test_period.drop(release_date_test_period.loc[pd.IndexSlice[:, [2017], ["Q1", "Q2", "Q3"]]].index)
release_date_test_period = release_date_test_period.drop(release_date_test_period.loc[pd.IndexSlice[:, [2020], ["Q4"]]].index)

# 各対象四半期の最終月(6月, 9月, 12月, 3月)に発表だと遅い。IBESは(6月1日, 9月1日, 12月1日, 3月1日)時点の予測で統一。
release_date_test_period.groupby(release_date_test_period.dt.month).count()

release_date_test_period[release_date_test_period.dt.month == 6]
release_date_test_period[release_date_test_period.dt.month == 9] # 理研ビタミン 2019  Q4  2020-09-30 <-- 1四半期遅れ
release_date_test_period[release_date_test_period.dt.month == 12]
release_date_test_period[release_date_test_period.dt.month == 3]

# report release date for each quarter
def get_late_firm(y, q):
    # print("/// month count ///")
    # print(release_date_test_period.loc[pd.IndexSlice[:, y, q]].groupby(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month).count())
    if q == "Q1":
        m1, m2 = 7, 8
    elif q == "Q2":
        m1, m2 = 10, 11
    elif q == "Q3":
        m1, m2 = 1, 2
    elif q == "Q4":
        m1, m2 = 4, 5
    # print("/// late firm ///")
    # print(release_date_test_period.loc[pd.IndexSlice[:, y, q]][~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m1) & ~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m2)])
    return release_date_test_period.loc[pd.IndexSlice[:, y, q]][~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m1) & ~(release_date_test_period.loc[pd.IndexSlice[:, y, q]].dt.month == m2)].index

late_firm = []
for i in [
    [2017, "Q4"], [2018, "Q1"], [2018, "Q2"], [2018, "Q3"], 
    [2018, "Q4"], [2019, "Q1"], [2019, "Q2"], [2019, "Q3"], 
    [2019, "Q4"], [2020, "Q1"], [2020, "Q2"], [2020, "Q3"]
    ]:
    late_firm += list(get_late_firm(i[0], i[1]))
late_firm = list(set(late_firm))

# drop late firm from y_hats_all
non_late_firm = [x for x in all_firm if (x not in late_firm)]
y_hats_all = y_hats_all.loc[pd.IndexSlice[non_late_firm, :, :], :] # 69 firms dropped because of 四半期決算発表日
# IBESは関係なく月末に予測がある企業を材料として使える?

# Price / FEPS ranking
stock_price = stock_price[["企業名", "end_month_date", "end_month_price"]]

def map_fiscal(x):
    if x.month == 5:
        y = x.year
        q = "Q1"
        v = [y, q]
    elif x.month == 8:
        y = x.year
        q = "Q2"
        v = [y, q]
    elif x.month == 11:
        y = x.year
        q = "Q3"
        v = [y, q]
    elif x.month == 2:
        y = x.year-1
        q = "Q4"
        v = [y, q]
    else:
        v = np.nan
    return v

stock_price["会計年度_四半期"] = stock_price["end_month_date"].apply(map_fiscal)
stock_price = stock_price.dropna()
stock_price["会計年度"] = stock_price["会計年度_四半期"].apply(lambda x: x[0])
stock_price["四半期"] = stock_price["会計年度_四半期"].apply(lambda x: x[1])
stock_price = stock_price.sort_values(by=["企業名", "会計年度", "四半期"], ascending=True)
stock_price = stock_price.set_index(["企業名", "会計年度", "四半期"])
stock_price = stock_price.loc[non_late_firm]

# 株価欠損値
[x for x in y_hats_all.index if (x not in stock_price.index)]

# マニュアル欠損補填
# https://96ut.com/stock/jikei.php?code=8108&year=2018
omit_price = pd.DataFrame([x for x in y_hats_all.index if (x not in stock_price.index)], columns=["企業名", "会計年度", "四半期"])
omit_price["end_month_date"] = ["2018-05-31", "2018-08-31", "2018-11-30", "2019-02-28", "2019-05-31", "2019-08-30", "2019-11-29", "2020-02-28", "2018-05-31", "2018-08-31", "2018-11-30"]
omit_price["end_month_date"] = pd.to_datetime(omit_price["end_month_date"])
omit_price["end_month_price"] = [1470, 1300, 1318, 1270, 1227, 1261, 1309, 1284, 2310, 2216, 2215]
omit_price = omit_price.set_index(["企業名", "会計年度", "四半期"])

stock_price = stock_price[["end_month_date", "end_month_price"]]

stock_price = pd.concat([stock_price, omit_price]).sort_index()
stock_price.loc["ヤマエ久野"]
stock_price.loc["未来工業"]

[x for x in y_hats_all.index if (x not in stock_price.index)]

# P / Earnings ratio
ratio = []
for e in y_hats_all.columns:
    rate = stock_price["end_month_price"] / y_hats_all[e]
    rate.name = "P/" + e
    ratio.append(rate)
ratio = pd.DataFrame(ratio).T

# PER --> RWと同じ

# Portfolio Strategy
def pe_portfolio(pe, start):
    pe = pe.loc[pd.IndexSlice[:, start[0], start[1]]]
    # exclude negative PER
    pe = pe[pe > 0]

    # High rate portfolio (short)
    p1 = pe.sort_values(ascending=False)[0 * (len(pe) // 5) : 1  * (len(pe) // 5)]

    p2 = pe.sort_values(ascending=False)[1 * (len(pe) // 5) : 2  * (len(pe) // 5)]
    p3 = pe.sort_values(ascending=False)[2 * (len(pe) // 5) : -2 * (len(pe) // 5)]
    p4 = pe.sort_values(ascending=False)[-2 * (len(pe) // 5) : -1 * (len(pe) // 5)]

    # Low rate portfolio (long)
    p5 = pe.sort_values(ascending=False)[-1 * (len(pe) // 5) : ]
    
    return [p1, p2, p3, p4, p5]

# each portfolio return
def portfolio_return(firm_name, start, end):
    buy_price = stock_price.loc[pd.IndexSlice[firm_name, start[0], start[1]], "end_month_price"].sum()
    sell_price = stock_price.loc[pd.IndexSlice[firm_name, end[0], end[1]], "end_month_price"].sum()
    r = (sell_price - buy_price) / buy_price
    # print("buy price: ", buy_price)
    # print("sell price: ", sell_price)
    # print("portfolio return: ", r)
    return r

pe = ratio[ratio.columns[4]]
start = [2018, "Q1"]
end = [2018, "Q2"]

ps = pe_portfolio(pe, start)
p1, p5 = ps[0].index, ps[-1].index

# p1 return
portfolio_return(p1, start, end)

# p5 return
portfolio_return(p5, start, end)

# Long p5 and Short p1
portfolio_return(p5, start, end) - portfolio_return(p1, start, end)

test_periods = [
    [2018, "Q1"], [2018, "Q2"], [2018, "Q3"], [2018, "Q4"],
    [2019, "Q1"], [2019, "Q2"], [2019, "Q3"], [2019, "Q4"],
    [2020, "Q1"], [2020, "Q2"], [2020, "Q3"], [2020, "Q4"]
    ]

# 1Q Hold return
df_pr = pd.DataFrame()
for j in ratio.columns:
    pe = ratio[j]
    print("model: ", pe)
    l = []
    for i in range(len(test_periods)-1):
        start = test_periods[i]
        end = test_periods[i + 1]
        r_hold_1q = list(map(lambda x: portfolio_return(x.index, start=start, end=end), pe_portfolio(pe, start)))
        r_hold_1q += [i]
        l.append(r_hold_1q)
    l = pd.DataFrame(l, columns=["P1", "P2", "P3", "P4", "P5", "t"])
    l["model"] = pe.name
    df_pr = pd.concat([df_pr, l], axis=0)
df_pr = df_pr.set_index(["model", "t"])

df_pr["Spread"] = df_pr["P5"] - df_pr["P1"]

# plt.figure(figsize=(16, 9))
# for i in df_pr.index.get_level_values(0).unique():
#     plt.plot(df_pr["Spread"].loc[i], marker=".", label=i)
#     plt.legend()
# plt.ylabel("Spread")
# plt.xlabel("t")
# plt.show()

# P / y_hat_IBES
[x for x in y_hat_ibes.index if (x not in y_hats_all.index)] # 9 firms dropped because of 四半期決算発表日
y_hat_ibes = y_hat_ibes.loc[[x for x in y_hat_ibes.index if (x in y_hats_all.index)]]

ratio_ibes = stock_price.loc[y_hat_ibes.index]["end_month_price"] / y_hat_ibes["y_hat_ibes"]
ratio_ibes.name = "P/y_hat_ibes"

# IBES portfolio
# 1Q Hold return

pe = ratio_ibes
print("model: ", pe)
l = []
for i in range(len(test_periods)-1):
    start = test_periods[i]
    end = test_periods[i + 1]
    r_hold_1q = list(map(lambda x: portfolio_return(x.index, start=start, end=end), pe_portfolio(pe, start)))
    r_hold_1q += [i]
    l.append(r_hold_1q)
df_pr_ibes = pd.DataFrame(l, columns=["P1", "P2", "P3", "P4", "P5", "t"])
df_pr_ibes["model"] = pe.name
df_pr_ibes = df_pr_ibes.set_index(["model", "t"])
df_pr_ibes["Spread"] = df_pr_ibes["P5"] - df_pr_ibes["P1"]

df_pr = pd.concat([df_pr, df_pr_ibes], axis=0)

print(df_pr.index.get_level_values(0).unique())

model_list = [
    'P/y_test', 'P/y_hat_rw', 'P/y_hat_srw', 'P/y_hat_sarima_f', 
    'P/y_hat_men_i_tuned_simple', 'P/y_hat_ml1_i_tuned_simple', 'P/y_hat_ibes'
    ]

plt.figure(figsize=(16, 9))
for i in model_list:
    plt.plot(df_pr["Spread"].loc[i], marker=".", label=i)
plt.legend()
plt.ylabel("Spread")
plt.xlabel("t")
plt.title("Long-Short Portfolio Spread (1Q Holding and Rebalance)")
plt.show()

print("return standard deviation: ")
print(df_pr["Spread"].loc[model_list].groupby(level=0).std())

plt.figure(figsize=(16, 9))
for i in model_list:
    plt.plot(df_pr["Spread"].loc[i].cumsum(), marker=".", label=i)
plt.legend()
plt.ylabel("Spread")
plt.xlabel("t")
plt.title("Long-Short Portfolio Cumulated Spread (1Q Holding and Rebalance)")
plt.show()