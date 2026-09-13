# NepGraph Data

## Dataset Overview
- **Source:** Historical price data from the Nepal Stock Exchange (NEPSE).
- **Frequency:** Daily closing prices.
- **Format:** CSV file (`nepse_prices.csv`) with dates as the index and stock tickers as columns.

## Data Processing & Cleaning
- **Missing Data:** Forward-filled to account for non-trading days or thinly traded stocks.
- **Filtering:** For any given time window, stocks with more than 10% missing (NaN) daily returns are excluded from the correlation network to ensure statistical validity.
- **Returns:** The network is constructed using Log Returns: $r_t = \ln(P_t / P_{t-1})$.

## Limitations & Caveats
- **Survivorship Bias:** The dataset relies on currently active or historically recorded symbols. Delisted stocks or merged entities may introduce survivorship bias in the backtesting pipeline.
- **Liquidity:** NEPSE is a frontier market. Many stocks may not trade every single day, leading to zero-return days which can artificially inflate or dampen correlation with highly liquid stocks.
