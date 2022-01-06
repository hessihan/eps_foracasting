# Master Thesis: Forecasting Japanese Corporate Earnings.

See the full paper (japanese) document.  
[full_paper](https://github.com/hessihan/eps_foracasting/blob/master/paper/tex/eps_forecast.pdf) (Please download and open the file in local. The preview in GitHub is creepy.)

Web appendixes are here.

Diebold-Mariano score distribution histograms  

* MAPE:  [dm_mape](https://github.com/hessihan/eps_foracasting/blob/master/paper/web_appendix/_dm_mat_MAPE.pdf)
* MSPE:  [dm_mspe](https://github.com/hessihan/eps_foracasting/blob/master/paper/web_appendix/_dm_mat_MSPE.pdf)

Accuracy scores by individual firms  

* MAPE:  [acc_by_firm_mape](https://github.com/hessihan/eps_foracasting/blob/master/paper/web_appendix/accuracy_by_firm_MAPE.pdf)
* MSPE:  [acc_by_firm_mspe](https://github.com/hessihan/eps_foracasting/blob/master/paper/web_appendix/accuracy_by_firm_MSPE.pdf)
* Large Forecast Error: [acc_by_firm_lfe](https://github.com/hessihan/eps_foracasting/blob/master/paper/web_appendix/accuracy_by_firm_LFE.pdf)

## Outline

---

Forecasting quarterly corporate earnings, such as Earnings Per Share (EPS) of Japanese firms.  

Compareing the forecast performance of classical time-series models (several types of SARIMA) and Machine learning approach (Ridge regression, LASSO regression, Elastic Net, Random forest regression and 3 layer MLP).  

Examing the forecasting power of exogenous variable including accounting data.  

Conducting Diebold-Mariano test to check the statistical significance of the difference between two forecast methods' accuracy.

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