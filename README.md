# Master Thesis: Forecasting Japanese Corporate Earnings.

See the full paper document.  
[full_paper](https://github.com/user/repo/blob/branch/other_file.md)

Web appendixes are here.
Diebold-Mariano score distribution histograms  
* MAPE  [dm_mape](https://github.com/user/repo/blob/branch/other_file.md)
* MSPE  [dm_mspe](https://github.com/user/repo/blob/branch/other_file.md)
Accuracy scores by individual firms  
* MAPE  [a_by_firm_mape](https://github.com/user/repo/blob/branch/other_file.md)
* MSPE  [a_by_firm_mspe](https://github.com/user/repo/blob/branch/other_file.md)
* Large Forecast Error [a_by_firm_lfe](https://github.com/user/repo/blob/branch/other_file.md)

## Outline

---

Forecasting quarterly corporate earnings, such as Earnings Per Share (EPS) of Japanese firms.  
Compareing the forecast performance of classical time-series models (several types of SARIMA) and Machine learning approach (Ridge regression, LASSO regression, Elastic Net, Random forest regression and 3 layer MLP).  
Examine the forecasting power of exogenous variable including accounting data.  
Conduct Diebold-Mariano test to check the statistical significance of the difference in two forecast methods' accuracy.

<!-- ## Additional research

---
* Creating dashboard interface using Python Dash.
* Binary (up or down) forecast.
* Comparing the performance of Model-based earning forecest (time-series, NN) with analysts and managers' forcast. (the problem is those forecasts are not quarterly recorded) -->

<!-- ## Model

---

* Random Walk (benchmark)
* SARIMAs
* Multi-layer Perceptron (MLP) 
* Long Short-Term Memory: LSTM

* "Univariate" or "Multivariate".
* "Expanding window" or "Rolling window" or just "No window, just test with one training" -->

## Data Sources

---

* Nikkei NEEDS FinancialQUEST

## Getting Started
```
git clone https://github.com/hessihan/eps_foracasting

#cd PATH_TO_EXECUTE_FILE
#python EXECUTE_FILE.py
```