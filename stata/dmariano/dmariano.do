import delimited "D:\0ngoing\thesis\repo\assets\y_hats\y_hats_all.csv", encoding(UTF-8)

gen quarter = mod(_n, 12)
replace quarter = 12 if quarter == 0

gen firm = 企業
encode firm, gen(nfirm)

summarize

// dmariano
// http://fmwww.bc.edu/RePEc/bocode/d/dmariano.html
// https://www.statalist.org/forums/forum/general-stata-discussion/general/1472634-running-diebold-mariano-test-using-panel-data

//
// xtset nfirm quarter
// forvalues i = 1(1)1089 {
// dmariano y_test y_hat_sarima_br y_hat_sarima_f if nfirm == `i'
// }

forvalues i = 1(1)1089 {
preserve
quietly keep  if nfirm == `i'
display as result "nfirm `i'"
quietly tsset quarter
dmariano y_test y_hat_sarima_f y_hat_mraf if nfirm == `i'
restore
}