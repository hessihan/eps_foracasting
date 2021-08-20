# Multivariate-Linear Model

# import external modules
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define MLM model class
class MLMModelList(object):
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    """
    
    The MLM model list for each prediction periods.
    
    MLM.1
    math::
        E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-4} + b_3 INV_{t-1} + b_4 AR_{t-1} + b_5 CAPX_{t-1} + b_6 GM_{t-1} + b_7 SA_{t-1} + b_8 ETR_{t-1} + b_9 LF_{t-1} + e_t
    
    MLM.2
    math::
        E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-4} + b_3 INV_{t-4} + b_4 AR_{t-4} + b_5 CAPX_{t-4} + b_6 GM_{t-4} + b_7 SA_{t-4} + b_8 ETR_{t-4} + b_9 LF_{t-4} + e_t

    Attributes
    ----------
    models : list
        list of `sm.tsa.statespace.SARIMAX()` objects to predict each periods.
    y_prim_train : pandas.Series
        primitive training data of y
    y_test : pandas.Series
        the remaining test data of y which is used to modify train data of y through window
    x_prim_train : pandas.Series
        primitive training data of x
    x_test : pandas.Series
        the remaining test data of x which is used to modify train data of x through window
    y_lag : list of int (1 ~ 4)
        number of lag for target variables.
    x_lag : list of int (1 ~ 4)
        number of lag for fundamental accounting variables.
    silent : bool (defualt True)
        False to allow outputting model summary and acf, pacf plot for model residual.
        
    """
    def __init__(self, y_prim_train, y_test, x_prim_train, x_test, y_lag, x_lag, silent=True, store_models=False):
        self.models = []
        self.y_prim_train = y_prim_train
        self.y_test = y_test
        self.x_prim_train = x_prim_train
        self.x_test = x_test
        self.y_lag = y_lag
        self.x_lag = x_lag
        self.silent = silent
        self.store_models = store_models
        
    def fit_rolling_window(self, window):
        """
        
        Iterate rolling window fitting and one period ahead forecasting for multivariate regression.
        
        Parameters
        ----------
        window: int
            Size of the rolling window.
            
        """
        # 一回 train_test_split してるのにもう一回統合してるのは非効率だけど
        # merge train and test for full period
        y_full = self.y_prim_train.append(self.y_test)
        afv_full = self.x_prim_train.append(self.x_test)

        # lag X (and lag y as explanatpry variable) # ここでyのラグをxに含めてx_fullとしている
#         x_full = pd.concat([y_full.shift(1), y_full.shift(4), x_full.shift(self.x_lag)], axis=1)
        x_full = pd.concat([y_full.shift(i) for i in self.y_lag] + [afv_full.shift(i) for i in self.x_lag], axis=1)
        x_full.columns = [y_full.name + "_lag" + str(i) for i in self.y_lag] + [j + "_lag" + str(k) for j in afv_full.columns for k in self.x_lag]
        
#         print("features: ", x_full.columns)

        y_pred = []
        pred_index = []
        for i in range(len(self.y_test)):
            # get temporary train index
            temp_train_index = self.y_prim_train.index.append(self.y_test.index[:i])
            # drop the head of training data
            temp_train_index = temp_train_index[-window:]
                        
            # slice temporary train data
            y_temp_train = y_full.loc[temp_train_index]
            x_temp_train = x_full.loc[temp_train_index]
            
#             print(y_temp_train.shape)
#             print(y_temp_train.index)
#             print(x_temp_train.shape)
#             print(x_temp_train.index)
            
            # drop nan
            anynan_index = x_temp_train[x_temp_train.isna().any(axis=1)].index
            x_temp_train.drop(anynan_index, inplace=True)
            y_temp_train = y_temp_train.drop(anynan_index)
            
            # define and estimate Linear regression model from sklearn
            reg = LinearRegression(fit_intercept=True).fit(x_temp_train.values, y_temp_train.values.reshape(-1, 1))
            
            # append fitted model to list --> 各 window を別々のpickle fileとして保存しないとメモリが足りない
            if self.store_models:
                self.models.append(reg)
                
            # get one step ahead test index
            one_step_ahead_test_index = self.y_test.index[i]
            pred_index.append(one_step_ahead_test_index)
            
            # predict one period ahead
            one_step_ahead_x_test = x_full.loc[one_step_ahead_test_index]
            
            # always predict one step ahead (row number len(y_train)+1 = 41, (40 start from 0))
            y_pred.append(reg.predict(one_step_ahead_x_test.values.reshape(1, -1)).item())
                
        pred_index = pd.MultiIndex.from_tuples(pred_index, names=self.y_test.index.names)
        y_hat = pd.Series(y_pred, index=pred_index, dtype=self.y_test.dtype)
        return y_hat
    
    def predict_window(self, ):
        """
        
        Return rolling window prediction of y from saved fitted model list.
        
        """
        y_hat = pd.Series([], dtype=test.dtype)
        # predict one period ahead
        for i in self.models:
            None
        return y_hat
    
# debug
if __name__ == "__main__":
    
    # import external libraries
    import sys
    import numpy as np
    import pandas as pd
    import pickle
    
    # import internal modules
    sys.path.insert(1, '../')
    from models.MLM import MLMModelList
    from utils.data_editor import train_test_split
    
    # Prepare Data
    
    # read processed data
    df = pd.read_csv("../../data/processed/tidy_df.csv", index_col=[0, 1, 2])
    
    i = df.index.get_level_values(0).unique().values[0]
    
    # y : "EPS"
    y = df.loc[pd.IndexSlice[i, :, :], "EPS"]

    # x, exogenous regressors : 'INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF'
    x = df.loc[pd.IndexSlice[i, :, :], ['INV', 'AR', 'CAPX', 'GM', 'SA', 'ETR', 'LF']]

    # time series train test split (4/5) : (1/5), yearly bases
    y_train, y_test = train_test_split(y, ratio=(4,1))
    x_train, x_test = train_test_split(x, ratio=(4,1))
    
    # MLM
    mlm = MLMModelList(
        y_prim_train=y_train,
        y_test=y_test,
        x_prim_train=x_train,
        x_test=x_test, 
        y_lag=[1, 2, 3, 4],
        x_lag=[1, 2, 3, 4],
        silent=True,
        store_models=False
    )
    mlm.fit_rolling_window(window=len(y_train)-4)