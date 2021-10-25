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
# y_hats_all.to_csv("../../assets/y_hats/y_hats_all_vsibes.csv")

# accuracy table
ind_name = ["Max_error", "Max_percentage_error", "MAE", "MAPE", "MSPE", "MAPE-UB", "MSPE-UB", "Large_error_rate"]
indicators = [Max_error, Max_percentage_error, MAE, MAPE, MSE, MAPEUB, MSPEUB, LargeErrorRate]
indicators = dict(zip(ind_name, indicators))

a = accuracy_table(y_hats_all["y_test"], y_hats_all, indicators).T
a.to_csv("../../assets/y_hats/accuracy_table_vsibes.csv")
a

ai = accuracy_table_i(y_hats_all["y_test"], y_hats_all, indicators)
# ai.to_csv("../../assets/y_hats/accuracy_table_i_vsibes.csv")
ai

