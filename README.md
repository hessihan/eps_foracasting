# Master Thesis: Forecasting Japanese Corporate Earnings.

## Outline

---

Forecasting quarterly corporate earnings, such as Earnings Per Share (EPS) of Japanese firms.  
Compareing the forecast performance of classical time-series models (several types of SARIMA) and deep learning approach (simple neural networks and LSTM).  
Examine the forecasting power of exogenous variable including accounting data.  
Variable selection from whole large accounting datasets (might be LASSO?).  

## Additional research

---
* Creating dashboard interface using Python Dash.
* Binary (up or down) forecast.
* Comparing the performance of Model-based earning forecest (time-series, NN) with analysts and managers' forcast. (the problem is those forecasts are not quarterly recorded)

## Model

---

* Random Walk (benchmark)
* SARIMA (classic statistical time series model)
* Vanila Neural Network
* Long Short-Term Memory: LSTM

* "Univariate" or "Multivariate".
* "Expanding window" or "Rolling window" or just "No window, just test with one training"

## Data Sources

---

* Nikkei NEEDS FinancialQUEST (small request limit)
* FSA EDINET (optionnal, only recent 5 years data recorded)

