import numpy as np
import pandas as pd
import datetime

df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
ind_cat = pd.read_excel("../../data/raw/FQ/CORPORATE/ind_cat_.xlsx")
adress = pd.read_excel("../../data/raw/FQ/CORPORATE/adress_.xlsx")

firms = df.index.get_level_values(0).unique()

ind_cat = ind_cat.drop([0, 1]).drop(["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"], axis=1).set_index("Unnamed: 0")
adress = adress.drop([0, 1]).drop(["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"], axis=1).set_index("Unnamed: 0")

miss_firms = ['エフ・ジェー・ネクスト', 'ココカラファイン', 'タケエイ', 'ビオフェルミン製薬', 'フルサト工業', 'マツモトキヨシホールディングス', 'ヤマエ久野', '京都きもの友禅', '前田建設工業', '前田道路', '協和エクシオ', '船井電機', '蛇の目ミシン工業']

firms = firms.drop(miss_firms)

ind_cat = ind_cat.loc[firms]
adress = adress.loc[firms]

ind_cat.to_csv("../../data/processed/ind_cat.csv")
adress.to_csv("../../data/processed/adress.csv")

accuracy_table_i = pd.read_csv("../../assets/y_hats/accuracy_table_i_2.csv", index_col=[0, 1])
accuracy_table_i = accuracy_table_i.loc[firms]

accuracy_table_i.loc[pd.IndexSlice[:, "Max_error"], "y_hat_sarima_br"]

a = pd.DataFrame()
for i in accuracy_table_i.index.get_level_values(1).unique():
    for j in accuracy_table_i.columns:
        s = accuracy_table_i.loc[pd.IndexSlice[:, i], j]
        s.index = accuracy_table_i.index.get_level_values(0).unique()
        s.name = j + "_" + i
        a = pd.concat([a, s], axis=1)
a = a.fillna(0)
a = a.loc[firms]

# a_i_features = pd.concat([a, ind_cat, adress], axis=1)

# utf-8 column
a["nikkei_cat_code_l"] = ind_cat["日経業種大分類"]
a["nikkei_cat_code_m"] = ind_cat["日経業種中分類"]
a["nikkei_cat_code_s"] = ind_cat["日経業種小分類"]
a["tse_cat_code"] = ind_cat["東証業種コード"]
a["jis_prefec_code_head"] = adress["登記上本店所在地‐地域コード"].apply(lambda x: x[:-4 + 1])
a["jis_prefec_code_main"] = adress["本社事務所所在地‐地域コード"].apply(lambda x: x[:-4 + 1])

def calculate_age(born):
    today = datetime.datetime(2021, 3, 31)
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

a["age_real_day"] = datetime.datetime(2021, 3, 31) - pd.to_datetime(adress["実質上設立年月日"])
a["age_real_day"] = a["age_real_day"].apply(lambda x: x.days)
a["age_real_year"] = pd.to_datetime(adress["実質上設立年月日"]).apply(calculate_age)
a["age_format_day"] = datetime.datetime(2021, 3, 31) - pd.to_datetime(adress["形式上設立年月日"])
a["age_format_day"] = a["age_format_day"].apply(lambda x: x.days)
a["age_format_year"] = pd.to_datetime(adress["形式上設立年月日"]).apply(calculate_age)

a.to_csv("../../stata/accuracy_features_i.csv")


# a_sample, add control (log market cap, year dummy) ... --> sample level data.