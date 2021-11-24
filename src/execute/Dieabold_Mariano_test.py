# Evaluate predicted value (y_hats) for each methods.
# import external packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

# import internal modules
sys.path.insert(1, '../')
from utils.accuracy import *
# https://github.com/johntwk/Diebold-Mariano-Test
from utils.dm_test import dm_test


def dm_table(y_hats, method_1, method_2, crit="MAPE"):
    firm_list = y_hats.index.get_level_values(0).unique()
    firm_list = list(firm_list)

    def dm_test_i(firm, crit=crit):
        firm_slice = y_hats.loc[pd.IndexSlice[firm, :, :], :]
        return dm_test(firm_slice["y_test"], firm_slice[method_1], firm_slice[method_2], h=1, crit=crit)

    dm_results = list(map(dm_test_i, firm_list))

    dm = pd.DataFrame(dm_results, index=firm_list)
    # ac_i = pd.read_csv("../assets/y_hats/accuracy_table_i.csv", index_col=[0, 1])
    # crit_i = ac_i.loc[pd.IndexSlice[:, crit], [method_1, method_2]]
    # crit_i.index = firm_list
    # dm = pd.concat([dm, crit_i], axis=1)
    def sig_level(x):
        sig_level = None
        if x <= 0.01:
            sig_level = "0.01"
        elif (0.01 < x) & (x <= 0.05):
            sig_level = "0.05"
        elif (0.05 < x) & (x <= 0.10):
            sig_level = "0.10"
        else:
            sig_level = "not stat sig"
        return sig_level
    dm["sig_level"] = dm["p_value"].apply(sig_level)
    return dm

def dm_stats_hist(method_1, method_2, y_hats, bins=35):
    dm_mape = dm_table(y_hats, method_1, method_2, crit="MAPE")

    # pivot and plot
    dm_mape.pivot(columns="sig_level", values="DM").plot.hist(bins=bins, figsize=(16, 9))
    plt.xlabel("DM stat")
    plt.ylabel("firm count")
    plt.legend()
    plt.title("DM-test (" + str(len(dm_mape)) + " firms) results: (1) " + method_1 + " vs. (2) " + method_2)
    plt.show()

    p_1n = (dm_mape[dm_mape["p_value"] <= 0.010]["DM"] < 0).sum()
    p1_5n = (dm_mape[(dm_mape["p_value"] <= 0.05) & (dm_mape["p_value"] > 0.01)]["DM"] < 0).sum()
    p5_10n = (dm_mape[(dm_mape["p_value"] <= 0.10) & (dm_mape["p_value"] > 0.05)]["DM"] < 0).sum()
    p10_n = (dm_mape[dm_mape["p_value"] > 0.10]["DM"] < 0).sum()
    p10_p = (dm_mape[dm_mape["p_value"] > 0.10]["DM"] > 0).sum()
    p5_10p = (dm_mape[(dm_mape["p_value"] <= 0.10) & (dm_mape["p_value"] > 0.05)]["DM"] > 0).sum()
    p1_5p = (dm_mape[(dm_mape["p_value"] <= 0.05) & (dm_mape["p_value"] > 0.01)]["DM"] > 0).sum()
    p_1p = (dm_mape[dm_mape["p_value"] <= 0.010]["DM"] > 0).sum()
    print(p_1n, p1_5n, p5_10n, p10_n, "|", p10_p, p5_10p, p1_5p, p_1p)

def dm_integrated_table(method_1, m2_list, y_hats, crit="MAPE"):
    # Create Table
    dm_test_p = pd.DataFrame()
    l = []
    for i in m2_list:
        method_2 = i
        pair = "(1) " + method_1 + " vs " + "(2) " + method_2

        # DM score for each firm
        dm_mape = dm_table(method_1=method_1, method_2=method_2, y_hats=y_hats, crit=crit)
        dm_mape["pair"] = pair
        dm_test_p = pd.concat([dm_test_p, dm_mape])

        # count firm for each statistic significance level
        p_1n = (dm_mape[dm_mape["p_value"] <= 0.010]["DM"] < 0).sum()
        p_5n = (dm_mape[dm_mape["p_value"] <= 0.050]["DM"] < 0).sum()
        p_10n = (dm_mape[dm_mape["p_value"] <= 0.100]["DM"] < 0).sum()
        # p_n = (dm_mape[dm_mape["p_value"] > 0.10]["DM"] < 0).sum()
        # p_p = (dm_mape[dm_mape["p_value"] > 0.10]["DM"] > 0).sum()
        p_10p = (dm_mape[dm_mape["p_value"] <= 0.100]["DM"] > 0).sum()
        p_5p = (dm_mape[dm_mape["p_value"] <= 0.050]["DM"] > 0).sum()
        p_1p = (dm_mape[dm_mape["p_value"] <= 0.010]["DM"] > 0).sum()
        # print(p_1n, p_5n, p_10n, p_n, "|", p_p, p_10p, p_5p, p_1p)
        # print(p_1n, p_5n, p_10n, "|", p_10p, p_5p, p_1p)
        l.append([pair, p_1n, p_5n, p_10n, p_10p, p_5p, p_1p])

    dm_test_p = dm_test_p.reset_index().set_index(["pair", "index"])

    dm_test_p_count = pd.DataFrame(l, columns=["pair", "p_1n", "p_5n", "p_10n", "p_10p", "p_5p", "p_1p"])
    dm_test_p_count = dm_test_p_count.set_index(["pair"])

    return dm_test_p, dm_test_p_count

def plot_heatmap(y_hats_all, dm_test_p_count, loss):
    num_firm = len(y_hats_all.index.get_level_values(0).unique())

    # plot count heatmap 
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))#, dpi=300)

    sns.heatmap(dm_test_p_count[["p_1n", "p_5n", "p_10n"]].values, cmap='Blues', ax=axs[0], vmax=None, vmin=None, annot=True, fmt="d", cbar=False, cbar_kws={})
    sns.heatmap(dm_test_p_count[["p_10p", "p_5p", "p_1p"]].values, cmap='Reds', ax=axs[1], vmax=None, vmin=None, annot=True, fmt="d", cbar=False, yticklabels=False)

    fig.suptitle("DM score (d=MAPE), firm counts (" + str(num_firm) + "firms) by significance level: method (1): " + m1)
    axs[0].set_title("# negative statistically significant firm")
    axs[0].set_xticklabels(labels=["p<0.01", "p<0.05", "p<0.10"], rotation=45, ha='right')
    axs[1].set_title("# positive statistically significant firm")
    axs[1].set_xticklabels(labels=["p<0.10", "p<0.05", "p<0.01"], rotation=45, ha='right')

    yticks = [x.split()[-1] for x in dm_test_p_count.index]
    axs[0].set_yticklabels(labels=yticks, rotation=0, ha='right')

    fig.subplots_adjust(wspace=0, hspace=0)

    plt.show()

    # plot count (percentage) heatmap 
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))#, dpi=300)

    sns.heatmap(dm_test_p_count[["p_1n", "p_5n", "p_10n"]].values * (1/num_firm) * 100, cmap='Blues', ax=axs[0], vmax=50, vmin=0, annot=True, fmt=".3f", cbar=False, cbar_kws={})
    sns.heatmap(dm_test_p_count[["p_10p", "p_5p", "p_1p"]].values * (1/num_firm) * 100, cmap='Reds', ax=axs[1], vmax=50, vmin=0, annot=True, fmt=".3f", cbar=False, yticklabels=False)

    fig.suptitle("DM score (loss: " + loss + "), firm counts (percentage /" + str(num_firm) + ") by significance level: method (1): " + m1)
    axs[0].set_title("% negative statistically significant firm")
    axs[0].set_xticklabels(labels=["p<0.01", "p<0.05", "p<0.10"], rotation=45, ha='right')
    axs[1].set_title("% positive statistically significant firm")
    axs[1].set_xticklabels(labels=["p<0.10", "p<0.05", "p<0.01"], rotation=45, ha='right')

    yticks = [x.split()[-1] for x in dm_test_p_count.index]
    axs[0].set_yticklabels(labels=yticks, rotation=0, ha='right')

    fig.subplots_adjust(wspace=0, hspace=0)

    plt.show()

# check
if __name__ == "__main__":

    # TS models
    y_hats_all = pd.read_csv("../../assets/y_hats/y_hats_all.csv", index_col=[0, 1, 2])

    loss = "MAD"
    
    # m1 = "y_hat_rw"
    m1 = "y_hat_sarima_br"
    # m1 = "y_hat_ml1_i_tuned_simple"
    
    dm_test_p, dm_test_p_count = dm_integrated_table(m1, y_hats_all.columns, y_hats_all, crit=loss)
    # dm_test_p.to_csv("../assets/DM_test_result/dm_test_p_" + m1 + ".csv")
    # dm_test_p_count.to_csv("../assets/DM_test_result/dm_test_p_count_" + m1 + ".csv")

    plot_heatmap(y_hats_all, dm_test_p_count, loss)

    # IBES
    y_hats_all_vsibes = pd.read_csv("../../assets/y_hats/y_hats_all_vsibes.csv", index_col=[0, 1, 2])

    loss = "MAPE"

    m1 = "y_hat_ibes"
    
    dm_test_p_ibes, dm_test_p_count_ibes = dm_integrated_table(m1, y_hats_all_vsibes.columns, y_hats_all_vsibes, crit=loss)
    # dm_test_p_ibes.to_csv("../assets/DM_test_result/dm_test_p_vsibes_" + m1 + ".csv")
    # dm_test_p_count_ibes.to_csv("../assets/DM_test_result/dm_test_p_count_vsibes_" + m1 + ".csv")

    plot_heatmap(y_hats_all_vsibes, dm_test_p_count_ibes, loss)