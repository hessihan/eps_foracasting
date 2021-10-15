import delimited "D:\0ngoing\thesis\repo\stata\reg\accuracy_features.csv", encoding(UTF-8) 

summarize

tab nikkei_cat_code_l
tab nikkei_cat_code_m
tab nikkei_cat_code_s
tab tse_cat_code
tab jis_prefec_code_head
tab jis_prefec_code_main

tabstat y_hat_sarima_f_mape, by(jis_prefec_code_head)
tabstat y_hat_sarima_f_mape, by(tse_cat_code)
tabstat y_hat_sarima_f_mapeub, by(jis_prefec_code_head)
tabstat y_hat_sarima_f_mapeub, by(tse_cat_code)

scatter y_hat_sarima_f_mape age_real_day
scatter y_hat_sarima_f_mape age_real_year

reg y_hat_sarima_f_mape age_real_day, robust
generate ln_age_real_day = ln(age_real_day)
reg y_hat_sarima_f_mape ln_age_real_day, robust
reg y_hat_sarima_f_mape age_format_day, robust
reg y_hat_sarima_f_mape age_real_year, robust
reg y_hat_sarima_f_mape age_format_year, robust

reg y_hat_sarima_f_mape ib3650.tse_cat_code, robust
reg y_hat_sarima_f_mape ib13.jis_prefec_code_head, robust

reg y_hat_sarima_f_mape ln_age_real_day ib3650.tse_cat_code ib13.jis_prefec_code_head, robust

// 東京ダミー

gen tokyo_dummy = 1 if jis_prefec_code_head == 13
replace tokyo_dummy = 0 if tokyo_dummy == . 

reg y_hat_sarima_f_mape ln_age_real_day ib3650.tse_cat_code i.tokyo_dummy, robust

// add control (log market cap, year dummy) ... --> sample level data.