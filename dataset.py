import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore")

# Dataset object
class Dataset():
    
    def __init__(self, file_path):
        # read csv as pandas dataframe
        self.data = pd.read_csv(file_path, header=0)
    
    def cleaning(self, nan_sign="-", cut_period=(2, 13)):
        """
        
        Cleaning raw data (delete headers, replace nan, cut half-year-recorded period, reindex).

        Parameters
        ----------
        nan_sign : value
            value which to be replaced with np.nan
        """
        self.data.drop([0, 1], axis=0, inplace=True)
        self.data.replace(nan_sign, np.nan, inplace=True)
        self.data.drop(range(2, 13), axis=0, inplace=True)
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

    def to_quarter(self, cum_col, mon_col=None):
        """
        
        Substituting cummulative values to get quarterly data.
        
        Parameters
        ----------
        cum_col : list
            cummulative and quaterly valued columns name tuple list
            [(［累計］, ［３ヵ月］), ...]
                    
        """
        # make sure not substituting on 1st quarter (June)
        # fill nan for 3 months data
        for i in cum_col:
            nan_index = self.data[i][self.data[i].isnull()].index.values[0]
            return nan_index
        
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

    def deseasonalize(self, ):
        """

        De-seasonalize time series data.

        """
        return None

    def standardize(self,)
        """

        Standardize into -1 ~ 1.

        """
        return None
    
    def build(self):
        """
        
        Building dataset with overall methods.
        
        """
        None

# Debugging
if __name__ == "__main__":
    
    # Define parameters preliminary
    file_path = "data/raw/FINFSTA_TOYOTA_199703_202004.csv"
    
    earning_v = ['売上高・営業収益', '売上総利益', '営業利益', '経常利益／税金等調整前当期純利益', '当期純利益（連結）', '１株当たり利益']
    account_v_bs = ['棚卸資産', '資本的支出', '期末従業員数', '受取手形・売掛金／売掛金及びその他の短期債権']
    account_v_pl = ['販売費及び一般管理費']
    
    num_col = ['決算月数'] + [i + '［累計］' for i in earning_v + account_v_pl] + [i + '［３ヵ月］' for i in earning_v + account_v_pl] + account_v_bs
    num_col = dict(zip(num_col, ["float64"] * len(num_col)))

    # Instanciate Dataset class
    dataset = Dataset(file_path)
    dataset.cleaning()
    dataset.change_data_types(date_col, num_col)

    
    dataset.data["決算期"].dt.month != 3