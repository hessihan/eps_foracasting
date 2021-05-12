# Data preprocessing module

import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore")

# Dataset object
class Dataset():
    
    def __init__(self, file_path):
        # read csv as pandas dataframe
        self.data = pd.read_csv(file_path, header=0)
    
    def cleaning(self, nan_sign="-"):
        """
        
        Cleaning raw data (delete headers, replace nan, cut half-year-recorded period, reindex).

        Parameters
        ----------
        nan_sign : value
            value which to be replaced with np.nan
        
        cut_period : tuple
            range of dropping period
        """
        self.data.drop([0, 1], axis=0, inplace=True)
        self.data.replace(nan_sign, np.nan, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
    def change_data_types(self, date_col, num_col):
        """
        
        Changing selected columns' data type.
            date colunm --> "datetime"
            numerical columns --> float 32 or 64? (whatever, will be changed for torch in NN estimates)

        Parameters
        ----------
        date_col : list
            columns name list which take datetime value
        
        num_col : list
            set of numerical columns name list and numerical data type
            
        """
        for i in date_col:
            self.data[i] = pd.to_datetime(self.data[i])
            
        self.data = self.data.astype(num_col)
        
    def cut(self, cut_period=(0, 11)):
        """
        
        cut half-year-recorded period, reindex.

        Parameters
        ----------
        cut_period : tuple
            range of dropping period
        """
        # store cut period
        # self.cut = self.data.iloc[: cut_period[1] + 1]
        
        self.data.drop(range(cut_period[0], cut_period[1]), axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def to_quarter(self, cum_col):
        """
        
        Substituting cummulative values to get quarterly data.
        # https://stackoverflow.com/questions/59092317/pandas-converting-annual-cumulative-values-to-quarterly-data
        # groupby.diff
        
        Parameters
        ----------
        cum_col : list
            cumulative columns name list
                    
        """
        self.data["年度"] = np.repeat(np.arange(self.data["決算期"].iloc[0].year, self.data["決算期"].iloc[-1].year), 4)
        
        for i in cum_col:
            # complete "［累計］" column by filling and cumsum "［３ヵ月］" columns
            self.data[i] = (self.data[i + "［累計］"].ffill() + 
                            (self.data[i + "［累計］"].isna() * self.data[i + "［３ヵ月］"]).fillna(0.0)
                           )
            # substract by the previous cumsum
            self.data[i] = (self.data.groupby("年度")
                            [i].diff() # substract by the previous cumsum
                            .fillna(self.data[i]) # fill the firstquarters with original column
                            )
        
    def fill_nan(self, col_name, method="itp"):
        """
        
        Filling missing values with specified method.

        Parameters
        ----------
        col_name : list
            selected time series columns name list
        
        method : {"mean", "LOCF", "itp"}
            the way how to fill missing values. 
            "mean" for filling with mean (`pd.fillna(pd.mean())`)
            "LOCF" for filling with last observation carried forward (`pd.ffill()`)
            "itp" for filling with interpolation (`df.interpolate()`)
        """
        for i in col_name:
            if method == "itp":
                self.data[i].interpolate(method='linear', inplace=True)
            elif method == "LOCF":
                self.data[i].ffill(inplace=True)
            elif method == "mean":
                self.data[i].fillna(pd.mean(self.data[i]), inplace=True)
            else:
                print("method not valid")
    
    def extract(self, col_name):
        """
        
        Extracting columns from whole dataset.

        Parameters
        ----------
        col_name : list
            selected columns name list
            
        """
        self.data = self.data[col_name]
        return None
    
    def detrend(self, ):
        """
        
        De-trend time series data.
        
        """
        return None

    def deseasonalize(self):
        """

        De-seasonalize time series data.

        """
        return None

    def trans(self, log=True, scale="norm", r=None):
        """

        Transforming and scaling time series data.
        
        Parameters
        ----------
        log : bool
            Log-transform
        
        scale : {"norm", "range", None}
            method of scaling.
            "norm" for normalization
            "range" for range-scaling
            None for not scaling
        
        r : tuple
            if scale == "range", provide an range tuple (min, max)

        """
        return None
    
    def build(self):
        """
        
        Building dataset with overall methods.
        
        """
        return None

# Executing data build
if __name__ == "__main__":
    
    # Define parameters preliminary
    file_path = "../../data/raw/FINFSTA_TOYOTA_199703_202004.csv"
    
    earning_v = ['売上高・営業収益', '売上総利益', '営業利益', '経常利益／税金等調整前当期純利益', '当期純利益（連結）', '１株当たり利益']
    account_v_bs = ['棚卸資産', '資本的支出', '期末従業員数', '受取手形・売掛金／売掛金及びその他の短期債権']
    account_v_pl = ['販売費及び一般管理費']
    
    num_col = ['決算月数'] + [i + '［累計］' for i in earning_v + account_v_pl] + [i + '［３ヵ月］' for i in earning_v + account_v_pl] + account_v_bs
    num_col = dict(zip(num_col, ["float64"] * len(num_col)))
    
    date_col = ['決算期', '決算発表日']
    
    

    # Instanciate Dataset class
    dataset = Dataset(file_path)
    dataset.cleaning()
    dataset.change_data_types(date_col, num_col)
    
    # fillnan with "itp" method columns (account_v_bs)
    dataset.fill_nan(account_v_bs, method="itp")
    
    # cut half-year periods
    dataset.cut(cut_period=(0, 11))
    
    # generate quarterly data by subtracting cumsum
    dataset.to_quarter(earning_v + account_v_pl)
    
    # fillnan with "itp" method columns (earning_v + account_v_pl)
    ####
    
    # select columns
    col = date_col + ["決算月数"] + [i for i in earning_v + account_v_pl] + account_v_bs
    dataset.extract(col)
    
    # save as csv
    dataset.data.to_csv("../../data/processed/dataset.csv")
    
