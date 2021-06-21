# 企業収益予測
1st Report

## 目的

## 予測目的

## データ処理

### 問題
* データの期間をどうするか?
    2000年以降から連結会計になっているから2000年以降?
    会計基準変更には注意するべき。かといって将来の会計基準変更を予測はできないから会計基準変更ダミーは意味ない?(lagなら予測できるか)
* 欠損値処理をどうするか?
    とりあえずkaggleの時系列予測マスターのように前後の平均で埋めた。
    でも、四半期で1年周期的な変数は前後平均だとかなり変になってしまう。
* 正規化するかどうか?
    今はlevelのままで予測しているけど、平均引いて分散で割ってデータを正規化してから予測してもいい。
    変動幅が1標準偏差 ~ 2標準偏差なら大きい変動と解釈できるやつ。(by菊池)
    企業間でそれぞれモデル推定するから、比較もできそう。
    予測値をlevelに直したいなら逆の操作(分散掛けて平均足せばいい)
    
### 今後の展開
* 区間予測
    今は点推定であまり多くのことを語ることができない。予測区間を出せたら、その区間を用いてなんかできそう（投資戦略とか）。<br>
    
    ブートストラップ?
    <br> 
    https://www.jstage.jst.go.jp/article/jscejcei/73/2/73_I_317/_article/-char/ja/<br> 
    https://logics-of-blue.com/time-series-analysis-by-nnet/<br> 
    <br>
    μ,σ^2の予測?
    https://aotamasaki.hatenablog.com/entry/2018/11/04/101400<br> 
    https://aotamasaki.hatenablog.com/entry/2019/03/01/185430<br> 
    <br>
    ベイズ? blitz LSTM pytorch
    https://qiita.com/qiita_kuru/items/8d20986b51c8e57e51b5<br> 
    https://ichi.pro/pytorch-de-no-beijian-lstm-pytorch-beijiandhi-pura-ninguraiburari-de-aru-blitz-o-shiyo-103491200821971<br> 
    https://towardsdatascience.com/bayesian-lstm-on-pytorch-with-blitz-a-pytorch-bayesian-deep-learning-library-5e1fec432ad3<br> 