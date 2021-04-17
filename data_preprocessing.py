# Execute Data Preprocessing
from src.data.build_dataset import Dataset

if __name__ == "__main__":
    # Define parameters preliminary
    file_path = "data/raw/FINFSTA_TOYOTA_199703_202004.csv"
    
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
    
    # output dataset
    dataset.data.to_csv("./data/processed/dataset.csv")