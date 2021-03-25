import numpy as np
import pandas as pd

import warnings

warnings.simplefilter("ignore")

# Dataset object
class Dataset():
    
    def __init__(self, file_path):
        # read csv as pandas dataframe
        self.data = pd.read_csv(file_path, header=0)
    
    def cleaning(self):
        """
        
        Cleaning raw data (delete headers, reindex, replace nan).
            
        """
        self.data.drop([0, 1], axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.data.replace("-", np.nan, inplace=True)
        
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
        Setting 3 columns for monthly values for quarterly --> how to deal with 決算発表日?
        
        Parameters
        ----------
        cum_col : list
            cummulative and quaterly valued columns name tuple list
            [(［累計］, ［３ヵ月］), ...]
        
        mon_col : list
            monthly valued columns name list
                    
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

    def build(self):
        """
        
        Building dataset with overall methods.
        
        """
        None

# Debugging
if __name__ == "__main__":
    
    file_path = "data/raw/FINFSTA_TOYOTA_199703_202004.csv"
    col_name = ['決算期', '決算月数', '決算発表日', 
                '売上高・営業収益［累計］', '売上総利益［累計］', '営業利益［累計］', '経常利益／税金等調整前当期純利益［累計］', '当期純利益（連結）［累計］', '１株当たり利益［累計］',
                '売上高・営業収益［３ヵ月］', '売上総利益［３ヵ月］', '営業利益［３ヵ月］', '経常利益／税金等調整前当期純利益［３ヵ月］', '当期純利益（連結）［３ヵ月］', '１株当たり利益［３ヵ月］',
                '棚卸資産', '資本的支出', '期末従業員数', '販売費及び一般管理費［累計］', '販売費及び一般管理費［３ヵ月］', '受取手形・売掛金／売掛金及びその他の短期債権']
    date_col = ['決算期', '決算発表日']
    num_col = {'決算月数': "int8", '売上高・営業収益［累計］': "float64", '売上総利益［累計］': "float64", 
               '営業利益［累計］': "float64", '経常利益／税金等調整前当期純利益［累計］': "float64", 
               '当期純利益（連結）［累計］': "float64", '１株当たり利益［累計］': "float64", 
               '棚卸資産': "float64", '資本的支出': "float64", '期末従業員数': "float64", 
               '販売費及び一般管理費［累計］': "float64", '受取手形・売掛金／売掛金及びその他の短期債権': "float64", 
               '売上高・営業収益［３ヵ月］': "float64", '売上総利益［３ヵ月］': "float64", '営業利益［３ヵ月］': "float64", 
               '経常利益／税金等調整前当期純利益［３ヵ月］': "float64", '当期純利益（連結）［３ヵ月］': "float64", '１株当たり利益［３ヵ月］': "float64", '販売費及び一般管理費［３ヵ月］': "float64"}
    
    dataset = Dataset(file_path)
    dataset.cleaning()
    dataset.change_data_types(date_col, num_col)