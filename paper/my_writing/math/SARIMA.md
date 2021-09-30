Box-Jenkins Model (SARIMA Model)

* SARIMA $(p, d, q) \times (P, D, Q)_s$

$$
    \Phi_p(B^s) \phi(B) (1 - B)^d = 
    \\
    where: \\
    \phi(B) = 1 - \phi_1B - \phi_2B^2 - \cdots - \phi_pB^p \\
    \Phi(B)
$$

* Brown-Rozeff (1979), $(1, 0, 0) \times (0, 1, 1)_4$ specification

$$
  E(Y_t) = Y_{t-4} + \phi_1 (Y_{t-1} - Y_{t-5}) - \Theta_1 a_{t-4} + \delta  
$$

* Griffin (1977), $(0, 1, 1) \times (0, 1, 1)_4$ specification

$$
  E(Y_t) = Y_{t-4} + (Y_{t-1} - Y_{t-5}) - \theta_1a_{t-4} - \Theta_1 a_{t-4} - \theta_1 \Theta_1a_{t-5} + \delta
$$

* Foster (1977), $(1, 0, 0) \times (0, 1, 0)_4$ specification

$$
  E(Y_t) = Y_{t-4} + \phi_1 (Y_{t-1} - Y_{t-5}) + \delta
$$