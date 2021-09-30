* 分析対象

本稿の分析対象サンプルは、2008年度以降から現在2020年度(2008/06 ~ 2021/03 の 52四半期)の間に東京証券取引所一部に上場し続けている3月決算の一般事業者(金融を除く)とする。2007年に企業会計基準委員会(ASBJ)が企業会計基準第12号「四半期財務諸表に関する会計基準」及び企業会計基準適用指針第14号「四半期財務諸表に関する会計基準の適用指針」を公表しており、本稿では四半期データを分析の単位としているため、分析期間を四半期報告制度が適用された2008年度(2008年4月1日以降)以降とした。最終的に各変数が計算可能なサンプルは13期×○○社=となった。

https://www.asb.or.jp/jp/accounting_standards/accounting_standards/y2007/2007-0314.html

* 被説明変数

$$\begin{eqnarray}
    Y &:& \text{四半期 1株あたり利益 (Quaterly EPS)} \\
    &=& \frac{普通株式に係る当期純利益}{普通株式の期中加重平均株式数} \\
    &=& \frac{損益計算書上の当期純利益-普通株式に帰属しない金額}{普通株式の期中加重平均発行済株式数 - 普通株式の期中加重平均自己株式数} \\
    \\
    普通株式の期中加重平均株式数 &=& 普通株式の期中加重平均発行済株式数 - 普通株式の期中加重平均自己株式数 \\
\end{eqnarray}$$

https://www.shinnihon.or.jp/corporate-accounting/commentary/other/2014-10-22.html

* 説明変数

ファンダメンタル会計変数, ファンダメンタルシグナル (fundamental accounting variables, fundamental signals)

Lav and Thiagarajan (1993), Abarbanell and Bushee (1997)

$$\begin{eqnarray}
    INV &:& \text{棚卸資産 (Inventory)} \\
    AR &:& \text{売掛金 (Accounts receivables)} \\
    CAPX &:& \text{Schedule V あたり資本的支出 (Capital expendituturep per Schedule V)} \\
    GM &:& \text{売上総利益 (Gross margin)} &=& \text{売上高} - \text{売上原価} \ (= \text{sales} - \text{cost of good sold}) \\
    SA &:& \text{販売費及び一般管理費(Selling and administrative expenses)} \\
    ETR &:& \text{実効税率 (Effective tax rate)} &=& \frac{法人税}{税引前利益} \\
    LF &:& \text{労働力 (Laborforce)} &=& \frac{売上高}{従業員数} \\
\end{eqnarray}$$

* 欠損処理

2011年に企業会計基準委員会(ASBJ)は改正企業会計基準第12号「四半期財務諸表に関する会計基準」及び改正企業会計基準適用指針第14号「四半期財務諸表に関する会計基準の適用指針」等の公表をしている。この四半期報告制度の改正では財務諸表作成者の作成負担を考慮し、いくつかの四半期情報の開示を義務ではなく任意としている。それに伴い、四半期報告制度の改正以降、資本的支出と期末従業員数の観測される頻度が減り、欠損が多く見られる。Zhang et al. (2004)は四半期単位で欠損している資本的支出と期末従業員数について、資本的支出は毎四半期均等であるとし、期末従業員数は前期の値を維持すると仮定して、欠損値を補填している。本稿でもこの方法に倣って資本的支出と期末従業員数の欠損処理を行うこととする。

https://www.asb.or.jp/jp/accounting_standards/accounting_standards/y2011/2011-0325.html

他の変数の欠損については原因を特定することが困難であったため、観測されている欠損値の前期と後期の平均値を欠損値に代入する。

* スケーリング処理

通常の$EPS$は当期純利益を普通株式(common shares)の期中(加重)平均株式数で割って算出される。そこでファンダメンタル会計変数のスケールを$EPS$のスケールと一致させるために変数$INV, AR, CAPX, GM, SA$は、$EPS$と同様に普通株式の期中(加重)平均株式数で割ってスケーリング処理を行う。$LF$は他の変数と比べてはるかに桁数が大きいため、対数変換によって他の変数と規模を揃える。

$$\begin{eqnarray}
    INV &=& \frac{棚卸資産}{普通株式の期中加重平均株式数} \\
    AR &=& \frac{売掛金}{普通株式の期中加重平均株式数} \\
    CAPX &=& \frac{資本的支出}{普通株式の期中加重平均株式数} \\
    GM &=& \frac{売上総利益}{普通株式の期中加重平均株式数} \\
    SA &=& \frac{販売費及び一般管理費}{普通株式の期中加重平均株式数} \\
    ETR &=& 実効税率 \\
    LF &=& \log{労働力} \\
\end{eqnarray}$$