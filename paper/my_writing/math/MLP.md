Univariate MLP

$$\begin{eqnarray}
    E(Y_t) &=& \sum_{j=1}^{n}\alpha_j \phi(\beta_{0j} + \sum_{i=1}^{k}\beta_{ij}X_i) \\
    &=& \sum_{j=1}^{n}\alpha_j \phi(\beta_{0j} + \sum_{i=1}^{4}\beta_{ij}Y_{t-i})
\end{eqnarray}$$

Multivariate MLP

$$\begin{eqnarray}
    E(Y_t) &=& \sum_{j=1}^{n}\alpha_j \phi(\beta_{0j} + \sum_{i=1}^{k}\beta_{ij}X_i) \\
    &=& \sum_{j=1}^{n}\alpha_j \phi(\beta_{0j} + \sum_{i=1}^{4}
    \left\{
    \beta_{1ij}Y_{t-i} + 
    \beta_{2ij}INV_{t-i} + 
    \beta_{3ij}AR_{t-i} + 
    \beta_{4ij}CAPX_{t-i} + 
    \beta_{5ij}GM_{t-i} + 
    \beta_{6ij}SA_{t-i} + 
    \beta_{7ij}ETR_{t-i} + 
    \beta_{8ij}LF_{t-i}
    \right\}
    )
\end{eqnarray}$$