import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# import internal modules
sys.path.insert(1, '../')
from utils.accuracy import *
from execute.accuracy_evaluation import *

y_hats_all = pd.read_csv("../../assets/y_hats/y_hats_all.csv", index_col=[0, 1, 2])
y_hat_ibes = pd.read_csv("../../data/processed/y_hat_ibes.csv", index_col=[0, 1, 2])

# limit to ibes firms
y_hats_all = y_hats_all.loc[y_hat_ibes.index]
y_hats_all["y_hat_ibes"] = y_hat_ibes
y_hats_all.to_csv("../../assets/y_hats/y_hats_all_vsibes.csv")

# accuracy table
ind_name = ["Max_error", "Max_percentage_error", "MAE", "MAPE", "MSPE", "MAPE-UB", "MSPE-UB", "Large_error_rate"]
indicators = [Max_error, Max_percentage_error, MAE, MAPE, MSE, MAPEUB, MSPEUB, LargeErrorRate]
indicators = dict(zip(ind_name, indicators))

a = accuracy_table(y_hats_all["y_test"], y_hats_all, indicators).T
a.to_csv("../../assets/y_hats/accuracy_table_vsibes.csv")
a

ai = accuracy_table_i(y_hats_all["y_test"], y_hats_all, indicators)
ai.to_csv("../../assets/y_hats/accuracy_table_i_vsibes.csv")
ai

# primal accuracy table for paper
y_test = y_hats_all["y_test"]
model_list = [
    'y_hat_rw', 
    'y_hat_srw', 
    'y_hat_sarima_f', 
    'y_hat_sarima_g', 
    'y_hat_sarima_br',
    'y_hat_mlm1', 
    'y_hat_mlm2',
    'y_hat_mlm4',
    'y_hat_ul1_i_tuned_simple',
    'y_hat_ul2_i_tuned_simple',
    'y_hat_uen_i_tuned_simple',        
    'y_hat_umlp',
    'y_hat_ml1_i_tuned_simple',
    'y_hat_ml2_i_tuned_simple',
    'y_hat_men_i_tuned_simple',
    'y_hat_mraf_i_tuned_simple',
    'y_hat_mmlp',
    'y_hat_ibes'
    ]
y_hat_list = list(map(lambda x: y_hats_all[x], model_list))
q_list = ["Q1", "Q2", "Q3", "Q4", ["Q1", "Q2", "Q3", "Q4"]]
score_list = [MAPEUB, MSPEUB, LargeErrorRate]    

a_by_q = []
for y_hat in y_hat_list:
    by_q = []
    for q in q_list:
            by_q.append(list(map(lambda s: s(y_test.loc[pd.IndexSlice[:, :, q]], y_hat.loc[pd.IndexSlice[:, :, q]]), score_list)))
    a_by_q.append(np.array(by_q).flatten())

a_by_q = pd.DataFrame(a_by_q)
a_by_q.index = model_list

col = [(i, j) for i in ["Q1", "Q2", "Q3", "Q4", "Overall"] for j in ["MAPE", "MSPE", "Large Forecast Error"]]
a_by_q.columns = pd.MultiIndex.from_tuples(col)
a_by_q.to_csv("../../assets/y_hats/accuracy_table_by_quarter_ibes.csv")
