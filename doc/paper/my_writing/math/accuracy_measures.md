予測の精度を測る指標

平均絶対パーセント誤差, 平均絶対誤差率 (Mean Absolute Percentage Error: MAPE)

$$
    \text{MAPE} = \frac{1}{T} \sum^{T}_{t=1} \left| \frac{Y_t - \hat{Y_t}}{Y_t} \right| \\
    (T: 訓練データの期間の長さ) \\
$$

平均二乗パーセント誤差, 平均二乗誤差率 (Mean Squared Percentage Error: MSPE)

$$
    \text{MSPE} = \frac{1}{T} \sum^{T}_{t=1} \left( \frac{Y_t - \hat{Y_t}}{Y_t} \right)^2
$$