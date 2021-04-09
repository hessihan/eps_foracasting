# Exploratory Data Analysis.

    # summarize data.
    # Why the quaterly data and diff-annual data different and which should be used?
    # What about missing value? Hou many missings?
    # How to deal with missing value?
    # Is there any trends or seanons in time series data?
    # ACF, PACF?

library(tidyverse)

eps <- read_csv("data/raw/FINFSTA_TOYOTA_199703_202004.csv")