# 人工ニューラルネットワークモデルによる日本企業の利益予測

早稲田大学商学研究科 修士課程2年 王思涵 35201064

## 1. イントロダクション

1株当たり利益（Earnings Per Share: EPS）は株式1つ当たりに対する企業の当期純利益を表した指標である。EPSは企業の当期純利益を発行株式数で割ることで求められており、規模に依存しない企業の収益性を示す。企業外部のステークホルダーである投資家は、企業の将来のEPSの予測をもとに、収益性の高いポートフォリオを構築したり、将来の株価収益率を算出して投資判断を行う。また、企業内部の経営者は将来のEPSの予測を用いて、営業予算の作成や設備投資の判断などの重要な意思決定を行う。したがって、EPSを正しく予測することは企業内外の幅広いステークホルダーにとって重要である。

企業の将来利益の予測は、人的な予測と、統計的・機械的な予測の2つに大別できる(桜井, 1990)。人的な予測は、予測をする主体によって、さらにアナリストによる予測と経営者による予測に分けられる。アナリスト予測とは、株式市場分析の専門家である証券アナリストが公表する利益予測であり、経営者予測は企業の内部者である経営者自らが公表する利益予測である(柴他, 2008)。他方、統計的・機械的な予測とは、過去の実績データをもとに何らかの時系列モデルを用いて将来の利益を予測することである。モデルに基づく予測は、予測値を導出するまでの過程を全て機械化できるため、人的な予測に比べてコストが低いという特徴を有している(桜井, 1990)。

統計的・機械的な予測に関する研究で用いられているモデルは、1)「単変量」か「多変量」か、2)「線形」か「非線形」か、の2つの観点で分類できる(Zhang et al., 2004)。従来の、モデルに基づくEPS予測の研究の多くは、自己回帰和分移動平均(AutoRegressive Integrated Moving Average: ARIMA)モデル(Box & Jenkins, 1977)などの単変量で線形な統計的時系列モデルを用いている(e.g., 英語のもっと古い論文)。一方、近年の研究では、「ファンダメンタル会計変数」を用いた多変量モデルによるEPS予測の研究が増えている。ファンダメンタル会計変数とは、売掛金、棚卸資産、資本的支出といったいくつかの会計変数のことであり、これらの変数には将来のEPSを予測する情報があるとされている(Lev and Thaigarajan, 1993; Abarbanell and Bushee, 1997)。また、EPSの財務データとしての非線形性を考慮するために、予測モデルに非線形性を反映させている研究も盛んである。中でもEPS予測の分野では、従来の統計的時系列モデルに代替されるモデルとして、人工ニューラルネットワーク(Artificial Neural Network: ANN)モデルが注目されている。ANNモデルは生物の神経細胞(ニューロン)の構造と機能をもとに考案された数学モデルであり、従来の統計モデルに比べて非線形で不定形な問題にうまく対処できることから、ビジネスの分野での利用が大きく増加している(Tkáč and Verner, 2016)。

ファンダメンタル会計変数を含めたANNモデルによる企業のEPS予測の研究の前例として、Zhang et al. (2004)はアメリカのニューヨーク証券取引所の上場企業、Cao and Gan (2009) は中国の上海証券取引所と深圳証券取引所の上場企業、Etemadi et al. (2015)はイランのテヘラン証券取引所の上場企業を対象に、EPSの予測を行っており、いずれの研究でもANNモデルが精度の高い予測をもたらす結果となっている。

このように近年、統計的・機械的な手法によるEPS予測の研究分野では、モデルに多変量と非線形な特性を反映させることで予測の精度が向上することが様々な国のサンプルで確認されている。しかし、太田 (2006)によるとアナリスト予測・経営者予測と従来の時系列モデル予測の精度比較研究については、明確な結論が出ていないにも拘らず、モデルによる予測よりも人的な予測の方が適切であると暗黙裡に見なされており、時系列モデルによる予測の研究は、現在では衰退しているとされている。現に日本における統計的・機械的なEPS予測の研究は限定的で、従来の線形時系列モデルの適用に留まっており、高い予測精度が期待されるANNモデルによる日本企業データを用いたEPS予測の研究は、筆者の知る限り存在しない。また、日本ではモデルによる利益予測が衰退したあとの2008年度に四半期報告制度ができているため、四半期データを用いたEPS予測の研究もない。

そこで本稿は、日本企業データにおいて代替的な時系列予測モデルであるANNモデルが、統計的・機械的な手法によるEPS予測の精度を向上させるのか、人的な予測よりも高い精度の予測をもたらすのかを検証する。具体的には東京証券取引所一部上場企業を対象にANNモデルによる四半期EPS予測を行い、得られた予測値を用いて従来の線形時系列モデルによる予測、アナリストによる予測、経営者による予測と精度比較をする。さらに、ANNモデルによる予測と各手法による予測の組み合わせ予測(Lobo & Nair, 1990; Conroy&Harris, 1987; Guerard, 1987)を求めて、ANNモデルによる予測に他の予測手法の精度を向上させる情報を含んでいるかどうかを調べる。追加的な検証として、ANNモデルによる予測値と各手法による予測値それぞれに基づいて、残余利益評価モデル(Residual Income Valuation Model: RIV)およびオールソンモデル・線形情報モデル(Linear Information Model: LIM)(Ohlson, 1995; Ohlson, 2001)を用いて企業の株式の理論価値を算出し、それを日本の証券市場に適用させて、ANNモデルによる予測と各手法による予測の価値関連性を考察する。

## 2. 過去の研究

これまで多くの利益予測の研究で、統計的・機械的な予測手法間の精度比較、統計的・機械的な予測手法と人的な予測手法の精度比較、人的な予測手法間の精度比較が行われている。桜井 (1990)や太田 (2006)のレビュー論文は、そのような数々の利益予測の研究に関する文献についてまとめている。アメリカの利益予測の研究に関して、桜井 (1990)は以下のようにまとめている。まずEPSを予測する伝統的な時系列モデルについて、年次EPSはランダムウォークモデルによってうまく描写され、四半期EPSはBrown-Rozeff (1979)、Griffin (1977)、Foster(1977)の3つのARIMAモデルによってうまく描写されるとしている。一方で、アナリスト予測と伝統的な時系列モデルによる予測の比較については、アナリストによる年次および四半期のEPS予測が伝統的な時系列モデルによる予測よりも正確であると述べている。その理由として、アナリストは広く最新な情報集合を予測に用いるからであるとしている。さらに、アメリカでは任意公表である経営者による年次の予測については、伝統的な時系列モデルやアナリスト予測よりも正確であるとしているが、任意に公表する企業の利益はもともと予測が容易な傾向があるため、全企業に対して経営者予測が最も正確であると一般化することはできないと指摘している。太田 (2006)では、アナリストと経営者予測の比較について、経営者予測はその公表前や公表時点のアナリスト予測よりも精度が高く、公表後ある一定期間を経過するとアナリスト予測の精度が経営者予測の精度を上回るとしている。また、日本の利益予測に関する研究については数が非常に少ないとしつつも、アメリカと同様な比較結果を得ているとしている。そして、時系列モデルによる予測に関しては、アナリスト予測との精度比較の明確な結論が出ていないにも拘わらず、最近の研究では時系列モデルによる予測よりもアナリスト予測を用いる方が市場の期待利益として適切であると暗黙裡に見なされており、時系列モデル予測の研究は、現在では衰退していると述べている。

ここで注意するべき点として、上記で言及されているEPS予測精度の比較の研究で扱われている時系列モデルの多くは、あくまでも当時研究が盛んであった伝統的な単変量で線形の時系列モデルのことであり、代替的な多変量・非線形モデルについては議論されていない。太田 (2006)は「ナイーブな時系列を用いた場合には経営者予測の方が精度が高いが、高度な時系列モデルを用いた場合には時系列モデルの予測の方が経営者予測よりも精度が高いといえる」と述べ、モデルの改善により統計的・機械的なEPS予測の精度は向上する余地があること示唆している。

近年、多くの学術分野で応用され成果を上げている予測モデルとして、人工ニューラルネットワーク(Artificial Neural Network: ANN)モデルが挙げられる。ビジネスに関する分野についても、監査・会計、財務分析、マーケティング、営業などの様々な文脈でANNモデルを応用した研究が行われている(Tkáč and Verner, 2016)。特に、Hill et al. (1996)は、月次や四半期の時系列予測においてANNモデルの方が伝統的な統計モデルよりも正確な予測を与えるとし、時系列予測におけるANNモデルの有用性を示している。その理由として、ANNはあらゆる関数形を近似できる普遍性定理(Universal Approximation Theorem)(Hornik et al., 1989)により、伝統的な線形統計モデルでは捉えられない時系列データの非線形性を捉え、予測モデルの関数形の誤特定を回避できるからであると述べている。また、Hill et al. (1994a)は、予測の対象である時系列データが、i)財務的、ii)季節的、iii)非線形な特徴を有するとき、ANNモデルによる予測は従来の線形時系列モデルよりも精度が高い傾向があると述べている。特に四半期EPSは上記の3つの特徴を有することも確認されている(Hopwood and McKeown, 1986; Lee and Chen, 1990; Callen et al, 1994)。

そこで、Callen et al. (1996)はニューヨーク株式市場の企業を対象に、単変量の四半期EPS予測におけるANNモデルと従来の統計的手法であるARIMAモデルとの精度を比較してANNの予測精度を検証したが、結果は予想に反してANNモデルによる予測がARIMAモデルによる予測よりも精度が低かった。そして、この研究ではANNの予測精度は文脈依存であると結論付けている。

一方、会計学の研究では、企業利益の予測について、将来の利益を説明する変数を特定することに注目してきた。Lav and Thiagarajan (1993)は、アナリストが有価証券の価値評価において有用であるとしている会計変数を調査し、それらの会計変数が将来の企業利益と関連があると述べている。言及された会計変数は以下のとおりである。

* 棚卸資産 (Inventories)
    売上原価の増加に対して過度な棚卸資産の増加は、売上高を増加させることが困難であることを示唆するため、ネガティブなシグナルとみなされる。さらに、棚卸資産が増加すると経営者は在庫を減らそうとするため、将来の利益が減少するシグナルになり得る。

* 売掛金 (Accounts receivable)
    売上高に対して過度な売掛金の増加は、企業の製品販売が困難な状態にあることや、賃倒引当金の増加などを意味し、将来の利益が減少するシグナルになり得る。

* 資本的支出 (Capital expenditures)
    資本的支出の過度な減少は、以前の投資水準を維持するための現在および将来のキャッシュフローが十分でないという経営者の懸念を意味する。このように一般的に資本的支出の減少は、短期的な経営志向とみなされ、将来の利益が減少するシグナルになり得る。

* 売上総利益 (Gross margin)
    売上総利益は企業の競争の激しさや営業レバレッジなどの要因を捉え、これが企業の長期的なパフォーマンスに影響を与える。したがって、売上総利益は企業の利益の持続性や企業価値に関して有益な情報をもち、売上高に対して過度な売上総利益の減少は、将来の利益が減少するシグナルになり得る。

* 販売費および一般管理費 (Selling and administrative expenses)
    ほとんどの場合、販売費および一般管理費は一定であるため、売上高に対して過度な販売費および一般管理費の増加は、コストコントロールの喪失や異常な販売努力を示唆し、将来の利益が減少するシグナルになり得る。

* 実効税率 (Effective tax rate)
    法定税率の変更に依らない企業の実効税率の大幅な変化は、一般的には一時的なものとして捉えられる。したがって、過度な実効税率の低下は将来の利益が減少するシグナルになり得る。

* 労働力 (Labor force)
    一般的に労働力の削減の発表に対してアナリストは好意的な反応を示す。したがって、過度な労働力の増加は将来の利益が減少するシグナルになり得る。

Abarbanell and Bushee (1997)では、上記の会計変数を「ファンダメンタル会計変数」とし、ファンダメンタル会計変数が将来のEPSの変化を説明するかどうかについて検証している。分析の結果、ファンダメンタル会計変数と将来のEPSの変化の間に強い関係が確認され、ファンダメンタル会計変数に含まれる会計情報は、将来のEPSに対して予測能力を持つと述べている。

これを受けZhang et al. (2004)は、Callen et al. (1996)の単変量ANNモデルによる四半期EPS予測の研究を発展させる形で、ANNモデルにファンダメンタル会計変数を反映させた多変量ANNモデルによる四半期EPS予測を行った。この研究では、単変量線形モデル、多変量線形モデル、単変量ANNモデル、多変量ANNモデルの4つの精度比較を行い、結果として、ANNモデルは線形モデルよりもうまくファンダメンタル会計変数の情報を反映し、より精度の高いEPSの予測値を与えた。その後、様々な国のサンプルでEPS予測におけるANNモデルの有用性を示した研究が行われている。Cao and Gan (2009) は中国の上海証券取引所と深圳証券取引所の上場企業、Etemadi et al. (2015)はイランのテヘラン証券取引所の上場企業を対象に、ファンダメンタル会計変数を反映させたANNモデルによるEPS予測を行っており、いずれも高い精度の予測を与える結果を得ている。

以上の過去の研究を踏まえ、現状の日本のEPS予測における「時系列モデルが古典的な手法にとどまっていること」と「時系列モデルと人的な予測の優劣の曖昧さ」の2つの課題に対し、本稿は近年多くのビジネスの分野で応用されているANNモデルにファンダメンタル会計変数を含めて日本企業の四半期EPSの予測を行い、さらにアナリスト、経営者、時系列モデルのEPS予測精度を比較し、予測手法間の精度の優劣を明らかにする。

## 参考文献

太田浩司 (2006)「経営者予想に関する日米の研究：文献サーベイ」『武蔵大学論集』第54巻、第1号、53-94頁。

桜井久勝 (1990)「会計利益の時系列特性と利益予測」『経営学・会計学・商学研究年報 / 神戸大学大学院経営学研究科編』第36号、45-98頁。

柴健次・薄井彰・須田一幸 (2008)『現代のディスクロージャー』中央経済社。

Abarbanell, J. S. and B. J. Bushee (1997) "Fundamental Analysis, Future Earnings, and Stock Prices", *Journal of Accounting Research*, Volume 35, Issue 1, pp.1-24.

Box, G. E. P., G. M. Jenkins, G. C. Reinsel, and G. M. Ljung (2016) *Time series analysis: Forecasting and control (5th edition)*, Hoboken, New Jersey: John Wiley and Sons, Inc.

Brown, L. D. and M. Rozeff (1979) "Univariate Time-Series Models of Quarterly Accounting Earnings per Share: A Proposed Model", *Journal of Accounting Research*, Volume 17, Issue 1, pp. 179-189.

Callen, J. L., C. C. Y. Kwan and P. C. Y. Yip (1994) "Non-linearity testing of quarterly accounting earnings", Working paper (Vincent C. Ross Institute, New York University). 

Callen, J. L., C. C. Y. Kwan, P. C. Y. Yip, and Y. Yuan (1996). "Neural network forecasting of quarterly accounting earnings", *International Journal of Forecasting*, Volume 12, Issue 4, pp. 475–482.

Cao, Q. and Q. Gan (2009) "Forecasting EPS of Chinese Listed Companies Using Neural Network with Genetic Algorithm", *15th Americas Conference on Information Systems 2009, AMCIS 2009*, Volumne 5, pp. 2971-2981.

Conroy, R. and R. Harris (1987) "Consensus forecasts of corporate earnings: analysts’ forecasts and time series methods", *Management Science*, Volume 33 Issue 6, pp. 725-738.

Etemadi. H., A. Ahmadpour and S. M. Moshashaei (2015) "Earnings Per Share Forecast Using Extracted Rules from Trained Neural Network by Genetic Algorithm", *Computational Economics*, Volumne 46, pp. 55-63.

Foster, G. (1977) "Quarterly Accounting Data: Time-Series Properties and Predictive-Ability Results", *The Accounting Review*, Volume 52, Issue 1, pp. 1-21.

Griffin, P (1977) "The time series behavior of quarterly earnings: Preliminary evidence", *Journal of Accounting Research*, Volume 15(Spring), pp. 71–83.

Guerard, J. B. (1987) "Linear constraints, robust-weighting and efficient composite modeling",  *Journal of Forecasting*, Volume 6, Issue 3, pp. 193–199.

Hill, T., M. O'Connor and W. Remus (1996) "Neural Network Models for Time Series Forecasts", *Management Science*, Volume 42, Issue 7, pp. 1082-1092.

Hill, T., L. Marquez, M. O'Conner and W. Remus (1994a) "Artificial neural network models for forecasting and decision making", *International Journal of Forecasting*, Volume 10, pp. 5-15. 

Hopwood, W. S. and J. C. McKeown (1986) *Univariate Time-series Analysis of Quarterly Earnings: Some Unresolved Issues, Studies in Accounting Research No. 25*, Sarasota, FL: American Accounting Association.

Hornik, K., M. Stinchcombe, and H. White (1989) "Multilayer Feedforward Networks are Universal Approximators", *Neural Networks*, Volume 2, Issue 5, pp. 359-366.

Lee, C. J. and C. Chen (1990) "Structural changes and fore- casting of quarterly accounting earnings in the utility industry", *Journal of Accounting and Economics*, Volume 13, pp. 93-122. 

Lev, B. and S. R. Thiagarajan (1993) "Fundamental Information Analysis", *Journal of Accounting Research*, Volume 31, Issue 2, pp. 190–215.

Lobo, G. J. and R. D. Nair (1990) "Combining judgmental and statistical forecasts: An application to earnings forecasts", *Decision Sciences*, Volumne 21, Issue 2, pp. 446-460.

Ohlson, J. (1995) "Earnings, Book Values, and Dividends in Equity Valuation", *Contemporary Accounting Research*, Volume 11, Issue 2, pp. 661-687.

Ohlson, J. (2001) "Earnings, Book Values, and Dividends in Equity Valuation: An Empirical Perspective", *Accounting Research*, Volume 18, Issue 1, pp. 107-120.

Tkáč, M. and R. Verner (2016) "Artificial neural networks in business: Two decades of research", *Applied Soft Computing*, Volumne 38, pp. 788-804.

Zhang, W., Q. Cao, and M. J. Schniederjans (2004) "Neural Network Earnings per Share Forecasting Models: A Comparative Analysis of Alternative Methods", *Decision Sciences*, Volume 35, Issue 2, pp. 205-237.