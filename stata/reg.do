import delimited "D:\0ngoing\thesis\repo\data\processed\tidy_df_lag.csv", encoding(UTF-8)

summarize

corr

* AbarbanellandBushee’s(1997)studyandthemodelused byLorekandWillinger(1996)
reg eps eps_1 eps_2 eps_3 eps_4 inv_1 ar_1 capx_1 gm_1 sa_1 etr_1 lf_1, robust

reg eps eps_1 eps_2 eps_3 eps_4 inv_4 ar_4 capx_4 gm_4 sa_4 etr_4 lf_4, robust

reg eps eps_1 eps_2 eps_3 eps_4 inv_1 inv_2 inv_3 inv_4 ar_1 ar_2 ar_3 ar_4 capx_1 capx_2 capx_3 capx_4 gm_1 gm_2 gm_3 gm_4 sa_1 sa_2 sa_3 sa_4 etr_1 etr_2 etr_3 etr_4 lf_1 lf_2 lf_3 lf_4 , robust

