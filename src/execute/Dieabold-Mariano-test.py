# Evaluate predicted value (y_hats) for each methods.
# import external packages
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# import internal modules
sys.path.insert(1, '../')
from utils.accuracy import *
from utils.dm_test import dm_test

# read y_hats
y_hats = pd.read_csv("../../assets/y_hats/y_hats_all.csv", index_col=[0, 1, 2])

# MAPE dm-test using existing package
def dm(y_hats, method1, method2, crit="MAPE"):
    firm_list = y_hats.index.get_level_values(0).unique()
    firm_list = list(firm_list)

    def dm_test_i(firm, crit=crit):
        firm_slice = y_hats.loc[pd.IndexSlice[firm, :, :], :]
        return dm_test(firm_slice["y_test"], firm_slice[method1], firm_slice[method2], h=1, crit=crit)

    dm_results = list(map(dm_test_i, firm_list))

    dm = pd.DataFrame(dm_results, index=firm_list)

    ac_i = pd.read_csv("../../assets/y_hats/accuracy_table_i_2.csv", index_col=[0, 1])
    crit_i = ac_i.loc[pd.IndexSlice[:, crit], [method1, method2]]
    crit_i.index = firm_list
    dm = pd.concat([dm, crit_i], axis=1)
    return dm

# check
dm_mape = dm(y_hats, "y_hat_rw", "y_hat_rw", crit="MAPE")
dm_mape[dm_mape["p_value"] <= 0.10].shape[0]
(dm_mape[dm_mape["p_value"] <= 0.10]["DM"] > 0).sum()
