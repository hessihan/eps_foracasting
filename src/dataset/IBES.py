# 2018/06/01における今期予測INT1MNが、全四半期の報告書発表後に更新された2018年Q1の予測のはず。(マニュアル的に2018/05/20 ~ 2018/05/31が良い?)
import os
import datetime
import numpy as np
import pandas as pd

# y_test
y_test = pd.read_csv("../../assets/y_hats/univariate/y_test.csv", index_col=[0, 1, 2])
test_firms = y_test.index.get_level_values(0).unique()

#### IBES
# concat IBES
# get all paths in selected directody

def read_ibes(dir_path, col):
    ibes = pd.DataFrame()
    for path in os.listdir(dir_path):
        # Define path
        data_file_path = dir_path + path
        # read temporary data file
        d = pd.read_excel(data_file_path)[col]
        est_date = path[:8]
        est_date = datetime.datetime(int(est_date[:4]), int(est_date[4:6]), int(est_date[6:]))
        d["est_date"] = est_date
        if est_date.month == 6:
            year = est_date.year
            quarter = "Q1"
        elif est_date.month == 9:
            year = est_date.year
            quarter = "Q2"
        elif est_date.month == 12:
            year = est_date.year
            quarter = "Q3"
        elif est_date.month == 3:
            year = est_date.year - 1
            quarter = "Q4"
        d["会計年度"] = year
        d["四半期"] = quarter
        # concat
        ibes = pd.concat([ibes, d], axis=0)
        print(path + " concated")

    ibes = ibes.set_index(["Type", "会計年度", "四半期"])
    return ibes

col = ['Type', 'NAME', 'INT1 MEAN EARN EST', 'INT1 END DATE']

dir_path1 = "../../data/raw/Datastream/IBES/1_1000/"
dir_path2 = "../../data/raw/Datastream/IBES/1001_2000/"
dir_path3 = "../../data/raw/Datastream/IBES/2001_/"

ibes_1 = read_ibes(dir_path1, col)
ibes_2 = read_ibes(dir_path2, col)
ibes_3 = read_ibes(dir_path3, col)

ibes = pd.concat([ibes_1, ibes_2, ibes_3])

# notna ratio
ibes["INT1 MEAN EARN EST"].notna().mean()

# notna firm ratio

# IBES NAME
ibes.index.get_level_values(0).nunique()

ibes.index.get_level_values(0).value_counts()
ibes.index.get_level_values(0).value_counts()[ibes.index.get_level_values(0).value_counts() > 12] # IBESの英名かぶり

enibes = ibes.index.get_level_values(0).unique()
enibes

# IBES code
code_dir = "../../data/raw/Datastream/IBES/code/"
ibes_code = pd.DataFrame()
for path in os.listdir(code_dir):
    print(path)
    d = pd.read_excel(code_dir + path)
    ibes_code = pd.concat([ibes_code, d])
    ibes_code.reset_index(drop=True, inplace=True)
ibes_code["code"] = ibes_code["LOC OFF. CODE"].apply(lambda x: int(x[1:]))

    
enibes_code = ibes_code[["code", "COMPANY NAME", "Type"]]
enibes_code.set_index("code", inplace=True)
enibes_code

#### FQ Firm NAME
# match firm name in JP and EN
en = pd.read_excel("../../data/raw/FQ/CORPORATE/EN.xlsx")
en = en.drop([0, 1])
en = en.reset_index(drop=True)
jp = pd.read_excel("../../data/raw/FQ/CORPORATE/JP.xlsx")
jp = jp.drop([0, 1])
jp = jp.reset_index(drop=True)

# EN firm name not matching JP name
en["Unnamed: 0"].nunique() == jp["Unnamed: 0"].nunique()
# code matched however
en["ISSUER CODE"].nunique() == jp["株式コード"].nunique()

# more than 4 sample
en["Unnamed: 0"].value_counts()[en["Unnamed: 0"].value_counts() > 4]
jp["Unnamed: 0"].value_counts()[jp["Unnamed: 0"].value_counts() > 4]

# less than 4 sample
en["Unnamed: 0"].value_counts()[en["Unnamed: 0"].value_counts() < 4]
jp["Unnamed: 0"].value_counts()[jp["Unnamed: 0"].value_counts() < 4]

overwrapped_name = en["Unnamed: 0"].value_counts()[en["Unnamed: 0"].value_counts() > 4].index

# for i in overwrapped_name:
#     print("")
#     print("###########################################################")
#     print(i)
#     print(en[en["Unnamed: 0"] == i])
#     print(jp.loc[en[en["Unnamed: 0"] == i].index])
    
# FQの英名かぶりすぎ問題

jp_code = jp[["Unnamed: 0", "株式コード"]].drop_duplicates()
jp_code.columns = ["JP", "code"]
jp_code["code"] = jp_code["code"].astype(int)
jp_code = jp_code.set_index("JP")

enfq_code = en[["Unnamed: 0", "ISSUER CODE"]].drop_duplicates()
enfq_code.columns = ["EN-FQ", "code"]
enfq_code["code"] = enfq_code["code"].astype(int)
enfq_code = enfq_code.set_index("code")

# jp.set_index("Unnamed: 0").loc[test_firms]
# KeyError: "['エフ・ジェー・ネクスト', 'ココカラファイン', 'タケエイ', 'ビオフェルミン製薬', 'フルサト工業', 'マツモトキヨシホールディングス', 'ヤマエ久野', '京都きもの友禅', '前田建設工業', '前田道路', '協和エクシオ', '船井電機', '蛇の目ミシン工業'] not in index" --> データ取得後、2021Q1, Q2で上場廃止や改名した企業。FQでしれっと消されてる。

omitted_name = {
    'エフ・ジェー・ネクスト': 8935, 
    'ココカラファイン': 3098, 
    'タケエイ': 2151, 
    'ビオフェルミン製薬': 4517, 
    'フルサト工業': 8087, 
    'マツモトキヨシホールディングス': 3088, 
    'ヤマエ久野': 8108, 
    '京都きもの友禅': 7615, 
    '前田建設工業': 1824, 
    '前田道路': 1883, 
    '協和エクシオ': 1951, 
    '船井電機': 6839, 
    '蛇の目ミシン工業': 6445
}

jp_code_omitted = pd.DataFrame(omitted_name.values(), index=omitted_name.keys(), columns=["code"])

# Create JP, EN, Code dataframe
name_code = pd.DataFrame(columns = ["JP", "EN-FQ", "EN-IBES", "code"])
# まず y_testの企業名を軸にcodeを追加
name_code["JP"] = test_firms
name_code = name_code.set_index("JP")
name_code["code"] = pd.concat([jp_code["code"], jp_code_omitted["code"]])
# 次に codeを軸にEN-FQ, EN-IBESを追加
name_code = name_code.reset_index()
name_code = name_code.set_index("code")
name_code["EN-FQ"] = enfq_code["EN-FQ"]
# name_code["EN-FQ"].isna().sum() # No changed name's recorded anymore in FQ
name_code["EN-IBES"] = enibes_code["COMPANY NAME"]
name_code["Type-IBES"] = enibes_code["Type"]

# FQのomitted企業のいくつかがEN-IBESでも欠損
name_code[name_code.isna().sum(axis=1) != 0]

##### output ibes y_hat
ibes = ibes.loc[name_code["Type-IBES"].dropna().values]
ibes.reset_index(inplace=True)
ibes["企業名"] = ibes["Type"].apply(lambda x: name_code["JP"][name_code["Type-IBES"] == x].values[0])
ibes.set_index(["企業名", "会計年度", "四半期"], inplace=True)

# EDA
# not nan value
ibes.notna().mean()
ibes[ibes["INT1 MEAN EARN EST"].notna()]

# full data firm
full_firm = ibes.index.get_level_values(0).unique()[
    (ibes["INT1 MEAN EARN EST"].notna().groupby(level=0).sum() == 12).values
]

y_hat_ibes = ibes.loc[full_firm]["INT1 MEAN EARN EST"]
y_hat_ibes.name = "y_hat_ibes"
y_hat_ibes.to_csv("../../data/processed/y_hat_ibes.csv")

