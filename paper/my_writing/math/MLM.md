## Multivariate-linear models

Abarbanell and Bushee (1997), Lorek and Willinger(1996)

MLM.1: Abarbanell and Bushee (1997), Lorek and Wikinger (1996), Zhang et al. (2004)

$$
    E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-4} + b_3 INV_{t-1} + b_4 AR_{t-1} + b_5 CAPX_{t-1} + b_6 GM_{t-1} + b_7 SA_{t-1} + b_8 ETR_{t-1} + b_9 LF_{t-1} + \epsilon_t
$$

MLM.2: Abarbanell and Bushee (1997), Lorek and Wikinger (1996), Zhang et al. (2004)

$$
    E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-4} + b_3 INV_{t-4} + b_4 AR_{t-4} + b_5 CAPX_{t-4} + b_6 GM_{t-4} + b_7 SA_{t-4} + b_8 ETR_{t-4} + b_9 LF_{t-4} + \epsilon_t
$$

MLM.3: Cao and Parry (2009)

$$
    E(Y_t) = a + b_1 Y_{t-1} + b_2 Y_{t-2} + b_3 Y_{t-3} + b_4 Y_{t-4} + b_5 INV_{t-4} + b_6 AR_{t-4} + b_7 CAPX_{t-4} + b_8 GM_{t-4} + b_9 SA_{t-4} + b_{10} ETR_{t-4} + b_{11} LF_{t-4} + \epsilon_t
$$

MLM.4: Original

$$\begin{eqnarray}
    E(Y_t) =
    && a + \\
    && b_1 Y_{t-1} + b_2 Y_{t-2} + b_3 Y_{t-3} + b_4 Y_{t-4} + \\
    && b_{5} INV_{t-4} + b_{6} INV_{t-4} + b_{7} INV_{t-4} + b_{8} INV_{t-4} + \\
    && b_{9} AR_{t-4} + b_{10} AR_{t-4} + b_{11} AR_{t-4} + b_{12} AR_{t-4} + \\
    && b_{13} CAPX_{t-4} + b_{14} CAPX_{t-4} + b_{15} CAPX_{t-4} + b_{16} CAPX_{t-4} + \\
    && b_{17} GM_{t-4} + b_{18} GM_{t-4} + b_{19} GM_{t-4} + b_{20} GM_{t-4} + \\
    && b_{21} SA_{t-4} + b_{22} SA_{t-4} + b_{23} SA_{t-4} + b_{24} SA_{t-4} + \\
    && b_{25} ETR_{t-4} + b_{26} ETR_{t-4} + b_{27} ETR_{t-4} + b_{28} ETR_{t-4} + \\
    && b_{29} LF_{t-4} + b_{30} LF_{t-4} +b_{31} LF_{t-4} +b_{32} LF_{t-4} +\\
    && \epsilon_t
\end{eqnarray}$$