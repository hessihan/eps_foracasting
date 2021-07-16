# external modules
import sys
import random
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.dates import DateFormatter
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm

# Pandas setting
# Display 6 columns for viewing purposes
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Reduce decimal points to 2
pd.options.display.float_format = '{:,.2f}'.format

# read temporary processed data
df = pd.read_csv("./../../data/processed/tse1_unitchanged.csv", index_col=[0, 1, 2])

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# * 2008Q1~ 6, 9, 12, 3月に四半期報告する企業に絞る
year_range = np.arange(2008, 2021)
month_range = [6, 9, 12, 3]

# check if firm recorded in all quater 
# start from firm recorded in 2008Q1
recorded_firm_from_2008 = set(df.loc[pd.IndexSlice[:, 2008, 6], :].index.get_level_values("企業名").unique())

for i in year_range:
    for j in month_range:
        if j == 3: # if Mar, +1 year
            i += 1
        recorded_firm_from_2008 = recorded_firm_from_2008 & set(df.loc[pd.IndexSlice[:, i, j], :].index.get_level_values("企業名").unique())

df = df.loc[pd.IndexSlice[recorded_firm_from_2008, :, :], :] # 遡るためrecorded_firm_from_2008の企業の全期間 : のデータフレームを作成

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# * multiindexを会計年度、四半期に変更
# change index to 会計年度, 四半期
df["会計年度"] = df["決算期"].apply(lambda x: int(x[:4])-1 if int(x[-2:]) <= 3 else int(x[:4])) # 1, 2, 3月なら西暦-1
df["四半期"] = df["決算期"].apply(lambda x: 
                            "Q1" if int(x[-2:]) ==  6 else (
                            "Q2" if int(x[-2:]) ==  9 else (
                            "Q3" if int(x[-2:]) == 12 else (
                            "Q4" if int(x[-2:]) ==  3 else (
                            "NonMar" + x[-2:]
                            ))))
                           )

# set multiindex ["企業名", "会計年度", "四半期"]
df.reset_index(inplace=True)
df.set_index(["企業名", "会計年度", "四半期"], inplace=True)

# * 2008Q1~ 3月本決算企業に絞る
# Q4 が 本決算である企業別割合が100%である企業
mar = (df.loc[pd.IndexSlice[:, 2008:, "Q4"], "決算種別フラグ"] == 10).mean(level=0)[
    (df.loc[pd.IndexSlice[:, 2008:, "Q4"], "決算種別フラグ"] == 10).mean(level=0) == 1
].index.get_level_values(0)

df = df.loc[pd.IndexSlice[mar, :, :], :]

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# df.to_csv("./../../data/processed/tidy_tse1_prefillna.csv")
# df = pd.read_csv("./../../data/processed/tidy_tse1_prefillna.csv", index_col=[0, 1, 2])

# 欠損処理
# * 補填カラムでベースカラムを補填 (累計を3ヶ月化)

# 期中平均株式数
# ベース '期中平均株式数［３ヵ月］'
# 補填 '期中平均株式数［累計］'
# 補填 '期末発行済株式総数'
    # 1株当たり平均を考慮して'期中平均株式数［累計］'で補填をする
    # それでも欠損している場合は'期末発行済株式総数'で補填する(株式数変更タイミングの場合が多い)
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop --> No drop
    # それでも欠損している場合は bfill
    
fillna_index = df[['期中平均株式数［３ヵ月］', '期中平均株式数［累計］']][
    (df['期中平均株式数［３ヵ月］'].isna()) & 
    (df['期中平均株式数［累計］'].notna()) & 
    (df['期中平均株式数［累計］'].shift(1).notna())
].index

# 累計で3ヶ月を補填
for i in fillna_index:
    if i[2] == "Q1":
        # MN_{Q1} = CMN_{Q1}
        df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［３ヵ月］'] = df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］']
    elif i[2] == "Q2":
        # MN_{Q2} = 2CMN_{Q2} - CMN_{Q1}
        df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［３ヵ月］'] = 2 * df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］'] - df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］'].shift(1)
    elif i[2] == "Q3":
        # MN_{Q3} = 3CMN_{Q3} - 2CMN_{Q2}
        df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［３ヵ月］'] = 3 * df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］'] - 2 * df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］'].shift(1)
    elif i[2] == "Q4":
        # MN_{Q4} = 4CMN_{Q4} - 3CMN_{Q3}
        df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［３ヵ月］'] = 4 * df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］'] - 3 * df.loc[pd.IndexSlice[i, :, :], '期中平均株式数［累計］'].shift(1)
    else:
        print("Invalid Quarter")
        pass

# '期末発行済株式総数'で残りの欠損を補填
fillna_index = df["期中平均株式数［３ヵ月］"][
    df["期中平均株式数［３ヵ月］"].isna()
].index

df.loc[pd.IndexSlice[fillna_index], "期中平均株式数［３ヵ月］"] = df.loc[pd.IndexSlice[fillna_index], "期末発行済株式総数"]

# それでも欠損してたらfillna(method="bfill")
df["期中平均株式数［３ヵ月］"] = df.groupby(level=0)["期中平均株式数［３ヵ月］"].fillna(method="bfill")

# df.loc[pd.IndexSlice[:, 2008:, :], "期中平均株式数［３ヵ月］"].isna().sum()
df.to_csv("./../../data/processed/tidy_tse1.csv")
df = pd.read_csv("./../../data/processed/tidy_tse1.csv", index_col=[0, 1, 2])

# 当期純利益
# ベース "当期純利益（連結）［累計］"
# 補填 "【ＱＴＲ】当期利益"
    # 新規カラム"当期純利益（連結）［３ヵ月］"を作成 (累計差分で3ヶ月に)
    # "【ＱＴＲ】当期利益"(多分値はちょっと違うけどいい近似にはなるでしょう)で補填する
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop --> No drop
    # それでも欠損している場合は interpolate

# 新規カラム"当期純利益（連結）［３ヵ月］"を作成 (累計差分で3ヶ月に)
df["当期純利益（連結）［３ヵ月］"] = df.groupby(level=[0, 1])['当期純利益（連結）［累計］'].diff().fillna(df['当期純利益（連結）［累計］'])

# "【ＱＴＲ】当期利益"(多分値はちょっと違うけどいい近似にはなるでしょう)で補填する
fillna_index = df["当期純利益（連結）［３ヵ月］"][
    df["当期純利益（連結）［３ヵ月］"].isna()
].index

df.loc[pd.IndexSlice[fillna_index], "当期純利益（連結）［３ヵ月］"] = df.loc[pd.IndexSlice[fillna_index], "【ＱＴＲ】当期利益"]

# それでも欠損している場合は interpolate
df["当期純利益（連結）［３ヵ月］"] = df["当期純利益（連結）［３ヵ月］"].interpolate(method="linear", limit=3) # groupbyしてないけど、多分大丈夫。一応確認した。

# df.loc[pd.IndexSlice[:, 2008:, :], "当期純利益（連結）［３ヵ月］"].isna().sum()

# １株当たり利益
# ベース '１株当たり利益［３ヵ月］'
# 補填 "１株当たり利益［累計］"
# 補填 "当期純利益（連結）［３ヵ月］" / '期中平均株式数［３ヵ月］'
    # "１株当たり利益［累計］"はQ1であるならQ1のみそのまま補填する
    # "当期純利益（連結）［３ヵ月］" / '期中平均株式数［３ヵ月］'で補填する
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate
    
# "１株当たり利益［累計］"はQ1であるなら補填する(多分補填できてるケースはない)
fillna_index = df.loc[pd.IndexSlice[:, :, "Q1"], "１株当たり利益［３ヵ月］"][
    df["１株当たり利益［３ヵ月］"].isna()
].index

df.loc[pd.IndexSlice[fillna_index], "１株当たり利益［３ヵ月］"] = df.loc[pd.IndexSlice[fillna_index], "１株当たり利益［累計］"]

# それでも欠損している場合は"当期純利益（連結）［３ヵ月］" / '期中平均株式数［３ヵ月］'で補填する
fillna_index = df["１株当たり利益［３ヵ月］"][
    df["１株当たり利益［３ヵ月］"].isna()
].index

df.loc[pd.IndexSlice[fillna_index], '１株当たり利益［３ヵ月］'] = df.loc[pd.IndexSlice[fillna_index], "当期純利益（連結）［３ヵ月］"] / df.loc[pd.IndexSlice[fillna_index], "期中平均株式数［３ヵ月］"]

# df.loc[pd.IndexSlice[:, 2008:, :], "１株当たり利益［３ヵ月］"].isna().sum()

# 棚卸資産
# ベース '棚卸資産'
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate(limit=4, limit_direction='both')

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], "棚卸資産"].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], "棚卸資産"].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# interpolate(method="linear", limit=3, limit_direction='both')
for i in df.index.get_level_values(0).unique():
    df.loc[pd.IndexSlice[i, :, :], "棚卸資産"] = df.loc[pd.IndexSlice[i, :, :], "棚卸資産"].interpolate(method="linear", limit=3, limit_direction='both')

# 売掛金
# ベース '受取手形・売掛金／売掛金及びその他の短期債権'
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], '受取手形・売掛金／売掛金及びその他の短期債権'].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], '受取手形・売掛金／売掛金及びその他の短期債権'].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# interpolate(method="linear", limit=3, limit_direction='both')
for i in df.index.get_level_values(0).unique():
    df.loc[pd.IndexSlice[i, :, :], '受取手形・売掛金／売掛金及びその他の短期債権'] = df.loc[pd.IndexSlice[i, :, :], '受取手形・売掛金／売掛金及びその他の短期債権'].interpolate(method="linear", limit=3, limit_direction='both')

# 資本的支出
# ベース '資本的支出'
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # 次期の値を問う分割して欠損に割り振り(or 0?)

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], '資本的支出'].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], '資本的支出'].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# 次期の値を問う分割して欠損に割り振り(or 0?)
for firm in df.index.get_level_values(0).unique():
    isna_series = df.loc[pd.IndexSlice[firm, :, :], "資本的支出"].isna()

    for i in range(len(isna_series)-1):
        if (isna_series.iloc[i] == True) & (isna_series.iloc[i+1] == False): 
            # i が nan, i+1 が nan じゃないとき、
            ind_non_nan = df.loc[pd.IndexSlice[firm, :, :], :].index[i+1]

            num_nan = 0
            ind_nan_list = []
            while isna_series.iloc[i - num_nan]:
                # nan がどのくらい続いていたのかカウントして、
                num_nan += 1
                ind_nan_list.append(df.loc[pd.IndexSlice[firm, :, :], "資本的支出"].index[i+1 - num_nan])
            # nan じゃない i+1 を num_nan + 1の分だけ割っておいて、
            divided_value = (df.loc[pd.IndexSlice[ind_non_nan[0], ind_non_nan[1], ind_non_nan[2]], "資本的支出"] / (num_nan+1))
            df.loc[pd.IndexSlice[ind_non_nan[0], ind_non_nan[1], ind_non_nan[2]], "資本的支出"] = divided_value

            for ind_nan in ind_nan_list:
                # nan にも同じ割った値を代入
                df.loc[pd.IndexSlice[ind_nan[0], ind_nan[1], ind_nan[2]], "資本的支出"] = divided_value
#     print(firm, " done")

# 売上総利益
# ベース '売上総利益［累計］'
    # 累計差分で3ヶ月化
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate

# 累計差分で3ヶ月化
df['売上総利益［３ヵ月］'] = df.groupby(level=[0, 1])['売上総利益［累計］'].diff().fillna(df['売上総利益［累計］'])

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], '売上総利益［３ヵ月］'].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], '売上総利益［３ヵ月］'].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# interpolate(method="linear", limit=3, limit_direction='both')
for i in df.index.get_level_values(0).unique():
    df.loc[pd.IndexSlice[i, :, :], '売上総利益［３ヵ月］'] = df.loc[pd.IndexSlice[i, :, :], '売上総利益［３ヵ月］'].interpolate(method="linear", limit=3, limit_direction='both')

# 販売費及び一般管理費
# ベース '販売費及び一般管理費［累計］'
    # 累計差分で3ヶ月化
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate

# 累計差分で3ヶ月化
df['販売費及び一般管理費［３ヵ月］'] = df.groupby(level=[0, 1])['販売費及び一般管理費［累計］'].diff().fillna(df['販売費及び一般管理費［累計］'])

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], '販売費及び一般管理費［３ヵ月］'].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], '販売費及び一般管理費［３ヵ月］'].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# interpolate(method="linear", limit=3, limit_direction='both')
for i in df.index.get_level_values(0).unique():
    df.loc[pd.IndexSlice[i, :, :], '販売費及び一般管理費［３ヵ月］'] = df.loc[pd.IndexSlice[i, :, :], '販売費及び一般管理費［３ヵ月］'].interpolate(method="linear", limit=3, limit_direction='both')

# 法人税
# ベース '法人税等［累計］'
    # 累計差分で3ヶ月化
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate

# 累計差分で3ヶ月化
df['法人税等［３ヵ月］'] = df.groupby(level=[0, 1])['法人税等［累計］'].diff().fillna(df['法人税等［累計］'])

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], '法人税等［３ヵ月］'].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], '法人税等［３ヵ月］'].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# interpolate(method="linear", limit=3, limit_direction='both')
for i in df.index.get_level_values(0).unique():
    df.loc[pd.IndexSlice[i, :, :], '法人税等［３ヵ月］'] = df.loc[pd.IndexSlice[i, :, :], '法人税等［３ヵ月］'].interpolate(method="linear", limit=3, limit_direction='both')
    
# 税引前利益
# ベース '税金等調整前当期純利益［累計］'
    # 累計差分で3ヶ月化
    # nan企業はすべてdrop

# 累計差分で3ヶ月化
df['税金等調整前当期純利益［３ヵ月］'] = df.groupby(level=[0, 1])['税金等調整前当期純利益［累計］'].diff().fillna(df['税金等調整前当期純利益［累計］'])

# nan企業はすべてdrop
nan_firms = df.loc[pd.IndexSlice[:, 2008:, :], '税金等調整前当期純利益［３ヵ月］'][
    df.loc[pd.IndexSlice[:, 2008:, :], '税金等調整前当期純利益［３ヵ月］'].isna()
].index.get_level_values(0).unique()

df = df.drop(nan_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# 売上高
# ベース '売上高・営業収益［累計］'
    # 累計差分で3ヶ月化
    # 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
    # interpolate

# 累計差分で3ヶ月化
df['売上高・営業収益［３ヵ月］'] = df.groupby(level=[0, 1])['売上高・営業収益［累計］'].diff().fillna(df['売上高・営業収益［累計］'])

# 2008Q1 ~ 2020Q4間で4連続(1年丸ごと)nanがある企業はdrop
whole_year_miss_firms = df.loc[pd.IndexSlice[:, 2008:, :], '売上高・営業収益［３ヵ月］'].isna().mean(level=[0, 1])[
    df.loc[pd.IndexSlice[:, 2008:, :], '売上高・営業収益［３ヵ月］'].isna().mean(level=[0, 1]) == 1
].index.get_level_values(0).unique()

df = df.drop(whole_year_miss_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

# interpolate(method="linear", limit=3, limit_direction='both')
for i in df.index.get_level_values(0).unique():
    df.loc[pd.IndexSlice[i, :, :], '売上高・営業収益［３ヵ月］'] = df.loc[pd.IndexSlice[i, :, :], '売上高・営業収益［３ヵ月］'].interpolate(method="linear", limit=3, limit_direction='both')
    
# 従業員数
    # fillna(method="ffill")

# fillna(method="ffill")
df["期末従業員数"] = df.groupby(level=0)["期末従業員数"].fillna(method='ffill')

# cutting off the data to 2008Q1 ~ after filling nan
# df.loc[pd.IndexSlice[:, 2008:, :], :].to_csv("./../../data/processed/tidy_tse1.csv")
# df = pd.read_csv("./../../data/processed/tidy_tse1.csv", index_col=[0, 1, 2])
# df.isna().sum()

# * 変数作成とスケーリング

# まずは"税金等調整前当期純利益［３ヵ月］" ==0(ETRがinfになる), "売上高・営業収益［３ヵ月］" < 0(log(-)になる)のレコードがある企業をdrop
infnan_firms = df[
    (df["税金等調整前当期純利益［３ヵ月］"] == 0) | 
    (df["売上高・営業収益［３ヵ月］"] < 0)
].index.get_level_values(0)

df = df.drop(infnan_firms, axis=0, level=0)

print("# firm: ", len(df.loc[pd.IndexSlice[:, 2008:, :], :].index.get_level_values(0).unique()))

df["EPS"] = df['１株当たり利益［３ヵ月］']
df["INV"] = df['棚卸資産'] / df['期中平均株式数［３ヵ月］']
df["AR"] = df['受取手形・売掛金／売掛金及びその他の短期債権'] / df['期中平均株式数［３ヵ月］']
df["CAPX"] = df['資本的支出'] / df['期中平均株式数［３ヵ月］']
df["GM"] = df['売上総利益［３ヵ月］'] / df['期中平均株式数［３ヵ月］']
df["SA"] = df['販売費及び一般管理費［３ヵ月］'] / df['期中平均株式数［３ヵ月］']
df["ETR"] = df["法人税等［３ヵ月］"] / df["税金等調整前当期純利益［３ヵ月］"]
df["LF"] = np.log(df["売上高・営業収益［３ヵ月］"] / df["期末従業員数"])

# select columns
df = df[['決算発表日', '事業年度開始年月日［３ヵ月］', '事業年度終了年月日', 
         'EPS', 'INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF']]

df.to_csv("./../../data/processed/tidy_df.csv")
df = pd.read_csv("./../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
