# Univariate Machine Learning Model
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(1, '../')
from utils.data_editor import train_test_split, lag
from utils.accuracy import *
from models.DataFlow import *

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from multiprocessing import Pool, cpu_count

def L1(alpha, x_train, y_train, x_test):
    model = Lasso(alpha=alpha, random_state=0)
    model.fit(x_train, y_train)
    y_hat = pd.Series(model.predict(x_test), index=x_test.index)
    return y_hat

def L2(alpha, x_train, y_train, x_test):
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(x_train, y_train)
    y_hat = pd.Series(model.predict(x_test), index=x_test.index)
    return y_hat

def EN(hyparams_1, hyparams_2, x_train, y_train, x_test):
    model = ElasticNet(alpha=hyparams_1, l1_ratio=hyparams_2)
    model.fit(x_train, y_train)
    y_hat = pd.Series(model.predict(x_test), index=x_test.index)
    return y_hat    

def RAF(hyparams_1, hyparams_2, x_train, y_train, x_test):  
    model = RandomForestRegressor(n_estimators=hyparams_1, max_depth=hyparams_2, max_features="auto", random_state=0)
    model.fit(x_train, y_train)
    y_hat = pd.Series(model.predict(x_test), index=x_test.index)
    return y_hat    

def MLP(hyparams_1, hyparams_2, x_train, y_train, x_test):
    model = MLPRegressor(
        hidden_layer_sizes=hyparams_1, 
        activation="logistic", 
        solver='adam', 
        alpha=hyparams_2, 
        # batch_size='auto', 
        # learning_rate="constant", 
        # learning_rate_init=0.001, 
        # power_t=0.5, 
        max_iter=1000, 
        # shuffle=True, 
        random_state=10, 
        tol=1e-6, 
        # verbose=False, 
        # warm_start=False, 
        # momentum=0.9, 
        # nesterovs_momentum=True, 
        early_stopping=True, 
        # validation_fraction=0.1, 
        # beta_1=0.9, 
        # beta_2=0.999, 
        # epsilon=1e-8, 
        # n_iter_no_change=10, 
        # max_fun=15000
        )
    model.fit(x_train, y_train)
    y_hat = pd.Series(model.predict(x_test), index=x_test.index)
    return y_hat

# PREPARE LAGGED DATA
my_df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
col = ['EPS'] # Univariate
#     col = ['EPS'] # univariate
my_df = my_df[col]
my_df = pd.concat([
    my_df,
    lagged_data(my_df, col, 1),
    lagged_data(my_df, col, 2),
    lagged_data(my_df, col, 3),
    lagged_data(my_df, col, 4),
], axis=1)
my_df.dropna(inplace=True)
my_firm_list = my_df.index.get_level_values(0).unique()
my_test_periods = [(i, j) for i in [2018, 2019, 2020] for j in ["Q1", "Q2", "Q3", "Q4"]]

y_test = pd.read_csv("../../assets/y_hats/univariate/y_test.csv", index_col=[0, 1, 2])

# TUNING
#     my_tune_space = np.linspace(0, 100, 101)
#     my_tune_space = np.meshgrid([0.001, 0.01, 0.1, 1, 10, 100, 1000])

# single
#     my_firm_list = my_firm_list[:2]
#     t1 = time.time()
#     y_hats = list(map(lambda firm: tune_i(my_df, my_firm_list, my_test_periods, 1, firm, 
#                                           L1, my_tune_space), tqdm(my_firm_list)))
#     t2 = time.time()
#     print(t2-t1)

# Multiprocessing
#     https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function
class Tuner_i(object):
    def __init__(self, df, firm_list, test_periods, val_size, method, tune_space):
        self.df = df
        self.firm_list = firm_list
        self.test_periods = test_periods
        self.val_size = val_size
        self.method = method
        self.tune_space = tune_space
    def __call__(self, i):
        return tune_i(self.df, self.firm_list, self.test_periods, self.val_size, i, self.method, self.tune_space)

# LASSO
my_tune_space = [[0.001], [0.01], [0.1], [1], [10], [100], [1000]]
t1 = time.time()
p = Pool(cpu_count() - 1)
y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, L1, my_tune_space), tqdm(my_firm_list)))
p.close()
t2 = time.time()
print(t2-t1)

name = "y_hat_ul1_i_tuned_simple"
y_hats = pd.concat(y_hats)
y_hats.index = y_test.index
y_hats = y_hats.rename(columns={"y_hat": name})
y_hats.to_csv("../../assets/y_hats/univariate/" + name + ".csv")
y_hats = pd.read_csv("../../assets/y_hats/univariate/" + name + ".csv", index_col=[0, 1, 2])
MAPEUB(y_test["y_test"].values, y_hats[name].values)

# fine tuning
my_tune_space = np.exp(np.linspace(-10, 10, 100)).reshape(-1, 1)
t1 = time.time()
p = Pool(cpu_count() - 1)
y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, L1, my_tune_space), tqdm(my_firm_list)))
p.close()
t2 = time.time()
print(t2-t1)

name = "y_hat_ul1_i_tuned_fine"
y_hats = pd.concat(y_hats)
y_hats.index = y_test.index
y_hats = y_hats.rename(columns={"y_hat": name})
y_hats.to_csv("../../assets/y_hats/univariate/" + name + ".csv")
y_hats = pd.read_csv("../../assets/y_hats/univariate/" + name + ".csv", index_col=[0, 1, 2])
MAPEUB(y_test["y_test"].values, y_hats[name].values)

# Ridge
my_tune_space = [[0.001], [0.01], [0.1], [1], [10], [100], [1000]]
t1 = time.time()
p = Pool(cpu_count() - 1)
y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, L2, my_tune_space), tqdm(my_firm_list)))
p.close()
t2 = time.time()
print(t2-t1)

name = "y_hat_ul2_i_tuned_simple"
y_hats = pd.concat(y_hats)
y_hats.index = y_test.index
y_hats = y_hats.rename(columns={"y_hat": name})
y_hats.to_csv("../../assets/y_hats/univariate/" + name + ".csv")
y_hats = pd.read_csv("../../assets/y_hats/univariate/" + name + ".csv", index_col=[0, 1, 2])
MAPEUB(y_test["y_test"].values, y_hats[name].values)

# EN
my_tune_space = np.vstack(map(np.ravel, np.meshgrid(
    [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
))).T
t1 = time.time()
p = Pool(cpu_count() - 1)
y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, EN, my_tune_space), tqdm(my_firm_list)))
p.close()
t2 = time.time()
print(t2-t1)

name = "y_hat_uen_i_tuned_simple"
y_hats = pd.concat(y_hats)
y_hats.index = y_test.index
y_hats = y_hats.rename(columns={"y_hat": name})
y_hats.to_csv("../../assets/y_hats/univariate/" + name + ".csv")
y_hats = pd.read_csv("../../assets/y_hats/univariate/" + name + ".csv", index_col=[0, 1, 2])
MAPEUB(y_test["y_test"].values, y_hats[name].values)

# RAF
my_tune_space = np.vstack(map(np.ravel, np.meshgrid(
    [100, 500, 1000, 2000],
    [10, 100, None]
))).T
t1 = time.time()
p = Pool(4) #Pool(cpu_count() - 1) # 熱暴走。ファンをあてよう
y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, RAF, my_tune_space), tqdm(my_firm_list)))
p.close()
t2 = time.time()
print(t2-t1)

name = "y_hat_uraf_i_tuned_simple"
y_hats = pd.concat(y_hats)
y_hats.index = y_test.index
y_hats = y_hats.rename(columns={"y_hat": name})
y_hats.to_csv("../../assets/y_hats/univariate/" + name + ".csv")
y_hats = pd.read_csv("../../assets/y_hats/univariate/" + name + ".csv", index_col=[0, 1, 2])
MAPEUB(y_test["y_test"].values, y_hats[name].values)

# MLP
my_tune_space = np.vstack(map(np.ravel, np.meshgrid(
    [tuple([4,]), tuple([8,]), tuple([16,])], # hidden_layer_sizes
    [1e-3, 1e-4, 1e-5] # alpha
))).T

my_tune_space = [
    [tuple([4,]), 1e-3],
    [tuple([8,]), 1e-3],
    [tuple([16,]), 1e-3],
    [tuple([4,]), 1e-4],
    [tuple([8,]), 1e-4],
    [tuple([16,]), 1e-4],
    [tuple([4,]), 1e-5],
    [tuple([8,]), 1e-5],
    [tuple([16,]), 1e-5],    
]

t1 = time.time()
p= Pool(cpu_count() - 1) # 熱暴走。ファンをあてよう
y_hats = list(p.map(Tuner_i(my_df, my_firm_list, my_test_periods, 1, MLP, my_tune_space), tqdm(my_firm_list)))
p.close()
t2 = time.time()
print(t2-t1)

name = "y_hat_umlp_sklearn_default"
y_hats = pd.concat(y_hats)
y_hats.index = y_test.index
y_hats = y_hats.rename(columns={"y_hat": name})
y_hats.to_csv("../../assets/y_hats/univariate/" + name + ".csv")
y_hats = pd.read_csv("../../assets/y_hats/univariate/" + name + ".csv", index_col=[0, 1, 2])
MAPEUB(y_test["y_test"].values, y_hats[name].values)