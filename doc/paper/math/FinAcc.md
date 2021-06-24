$$\begin{eqnarray}
\end{eqnarray}$$

$$\begin{align}
& a + b = 0 \\
& 2 = 0 \\
\end{align}$$

## 配当割引モデル (DDM: Dividend Discount Model)
株式の本質的価値(内在価値, 理論株価, intrinsic value)は将来支払われる配当の現在価値の合計とする株式評価モデル

$$\begin{eqnarray}
    V_0 &=& \frac {E[DIV_1]} {1 + r_e} + \frac {E[DIV_2]} {(1 + r_e)^2} + \frac {E[DIV_3]} {(1 + r_e)^3} +  \cdots \\
    &=& \sum^{\infty}_{t=1} \frac {E[DIV_t]} {(1 + r_e)^t}\\
    V_0 &:& \text{0期末(評価時点)における株式の本質的価値} \\
    DIV_t &:& \text{t期末(将来)の配当金の期待値} \\
    r_e &:& \text{株主資本コスト(株主が現時点において将来にわたって要求する収益率)} \\
\end{eqnarray}$$

無限の期間に渡って配当を予測することは不可能である。そこで評価時点から一定期間までの予測期間$T$を設定し、予測期間以降の配当を単純な仮定で代入する。例えば、、、、-->これはRIMのところで話せばいいか。

## クリーン・サープラス関係の仮定

$$\begin{eqnarray}
    \Delta BV_t = BV_{t} - BV_{t-1} =  NI_{t} - DIV_{t} \tag{2}
\end{eqnarray}$$

## 残余利益モデル (RIM: Residual Income Model)

株式の本質的価値$V^{RIM}_{0}$

$$\begin{eqnarray}
    RI_t &=& NI_t - r_e BV_{t-1} \\
    V^{RIM}_{0} &=& BV_0 + \frac {E[RI_1]} {1 + r_e} + \frac {E[RI_2]} {(1 + r_e)^2} + \frac {E[RI_3]} {(1 + r_e)^3} +  \cdots \\    
    &=& BV_0 + \sum^{\infty}_{t=1} \frac {E[RI_t]} {(1 + r_e)^t} \\
    &=& BV_0 + \sum^{\infty}_{t=1} \frac {E[NI_t - r_e BV_{t-1}]} {(1 + r_e)^t} \tag{*} \\
    &=& BV_0 + \sum^{\infty}_{t=1} \frac {E[(ROE_t - r_e) BV_{t-1}]} {(1 + r_e)^t} \tag{*} \\
    RI_t &:& \text{t期の残余利益} \\
    NI_t &:& \text{当期純利益(クリーンサープラス関係が成り立つならt期の純資産の増減} \Delta BV_t と一致) \\
    BV_t &:& \text{t期末における純資産簿価} \\
    ROE_t &:& \text{tの株主資本利益率} = \frac {NI_t} {BV_t} \\
\end{eqnarray}$$

ちなみに両辺を発行済み株式数で割ると、$V^{RIM'}_{0}$を株式の理論価格として解釈できる。

$$\begin{eqnarray}
    V^{RIM'}_{0}  
    &=& BPS_0 + \sum^{\infty}_{t=1} \frac {E[RI_t]} {(1 + r_e)^t} \\
    &=& BPS_0 + \sum^{\infty}_{t=1} \frac {E[EPS_t - r_e BPS_{t-1}]} {(1 + r_e)^t} \\
    &=& BPS_0 + \sum^{\infty}_{t=1} \frac {E[(ROE_t - r_e) BPS_{t-1}]} {(1 + r_e)^t} \\
    EPS_t &:& \text{1株当たり当期純利益} = \frac{NI_t} {発行済み株式総数} \\
    BPS_t &:& \text{1株当たり純資産} \\
\end{eqnarray}$$

## CAPM

$$\begin{eqnarray}
    r_{i, t}  - r_{f, t} &=& \beta_i (r_{m, t} - r_{f, t}) \\
    r_{i, t} &:& \text{企業iのt期末におけるリターン} \\
    r_{f, t} &:& \text{無リスク利子率, リスクフリーレート (10年物国債利回り)} \\
    r_{m, t} &:& \text{株価指数 (TOPIX)}\\
\end{eqnarray}$$

実際の推定は、以下の回帰モデルを用いる。

$$\begin{eqnarray}
    r_{i, M} - r_{f, M} &=& \alpha_i + \beta_i (r_{m, M} - r_{f, M}) + \epsilon_{i, M} \\
    r_{i, M} &:& \text{企業iの月次Mにおけるリターン} \\
    r_{f, M} &:& \\
    r_{m, M} &:& \\
\end{eqnarray}$$

推定後、$\hat \beta_i$を得て、$\hat \alpha_i=0$を確認したうえで、説明変数の平均値をモデルに代入し企業別株主資本コスト$r_{ei}$を算出する。

$$\begin{eqnarray}
    r_{ei} &=& \beta_i (\bar{r}_{m} - \bar{r}_{f}) + \bar{r}_{f}\\
    \bar{r}_{m} &=& \frac {1} {T} \sum^{T}_{M=1} r_{m, M} \\
    \bar{r}_{f} &=& \frac {1} {T} \sum^{T}_{M=1} r_{f, M} \\
\end{eqnarray}$$
## Fama-French 3 factor model (Fama and French, 1997)

$$\begin{eqnarray}
    r_{i, t}  - r_{f, t} &=& \beta_i (r_{m, t} - r_{f, t}) + + \\
\end{eqnarray}$$

https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A1%E3%83%BC%E3%83%9E-%E3%83%95%E3%83%AC%E3%83%B3%E3%83%81%E3%81%AE3%E3%83%95%E3%82%A1%E3%82%AF%E3%82%BF%E3%83%BC%E3%83%A2%E3%83%87%E3%83%AB
https://www.investopedia.com/terms/f/famaandfrenchthreefactormodel.asp

## Conditional 3 factor model




連結ver  
    時系列予測  
        y  
        
        * ROE
        * EPS (ROEの代わりに)
        * BV (経営者予想と合わせるならNI, DIV?)
        
        x  
    CAPM, Fama-French 3 Factor
        * R_{i,t} : 企業iの月次Mにおけるリターン
        * R_{f,t} : 無リスク利子率(リスクフリーレート), 10年物国債利回り
        * R_{m,t} : 株価指数, TOPIX
        * R_{m,t} : 株価指数, 日経平均
    RIM
        * BV_t : (t期末の)純資産
        * NI_t : 当期純利益
        * FNI_{t+1} : 当期純利益の1期先予測
        * FDIV_{t+1} : 配当の1期先予測
            --> FBV_{t+1} = BV_t + FNI_{t+1} + FDIV_{t+1} : 純資産の1期先予測

        * BPS_t : 1株当たり純資産
            --> FBPS_{t+1} : 1株当たり純資産の1期先予測
            1株当たりでやるならこれ。どっちにしろ Pを株価とするか時価総額とするかで求められる。
            
        * ROE_t : 株主資本利益率 (NI/BV)
        * FROE_{t+1} : ROEの1期先予測
        * FROE_{t+2} : ROEの2期先予測
        
        * EPS_t : 1株あたり利益
        * FEPS_{t+1} : EPSの1期先予測
        * FEPS_{t+2} : EPSの2期先予測
            ROEの代わりにEPS使えるけど、みんなROEっぽい
        
        * R_e : 株主資本コスト
            <-- CAPMや3factorで算出する(推定後、説明変数の平均値を代入したy_hatにx12して年次換算(もしくはx4して四半期率換算)してR_eを算出)
    VRP
        P_i : (RIMでBPS使うなら)6月末時点株価, (RIMでBV使うなら)6月末時点時価総額
            --> R_{i,t}を算出するなら株価はすでに入手済みなはず。
    
個別ver
