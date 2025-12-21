# cvx-trader

This repository is used for a trading bot that applies convex optimization to make decisions, via the [CVXPY](https://www.cvxpy.org/) library developed at Stanford.

As of now, it trades on fake money using an Alpaca Paper account that started with 100K dollars.

### Stock Trader Iterations

Version 1: simple convex optimization with quadratic programming and data smoothing with EWMA and Ledoit-Wolf covariance matrix. 

Version 2: more robust convex least-squares objective rather than quadratic form, improved numerical stabilty with Cholesky Factorization. After some parameter tuning, running backtest from 2023/01/01 to September 19, 2025 yields an annualized return of 34.35%, volatility of 12.34%, and a nice Sharpe ratio of 2.78.

Version 3: new logic to detect market shifts for dynamic risk aversion. Hidden Markov Models are under trial, k-means clustering works for now.

Future: automated parameter tuning, options trading with Black-Scholes.

### Helper Files

The script `plot_risk_return.py` assists in gauging the annualized risk-return trade-off between various assets. When plotting, one can observe that higher returns often lead to higher risk.

The Python script `purge_helper.py` is for helping to get rid of any shares with fractional value on Alpaca.

## Trading Strategy

The strategy used in the trading algorithm is mean-variance optimization. In any investment, we want to maximize gain with the least amount of risk within a certain trading period.

Judging an asset's ability to increase as well as its risk can be based on historical data, as well as current signals. Unsurprisingly, the patterns of the market vary over time. It is also important to account for the fact that future returns may not look like past returns at all.

For example, the picture below is an annualized risk-return plot for a large number of assets from January 2024 to September 2025. Notice that higher gain tends to come with more risk, although there are some exceptions. The position of assets on the plot will also highly depend on the time frame examined. Judging an asset's performance from the last year will yield different results than looking at the last month.

From the various market signals, the algorithm simply determines portfolio allocation to various assets. If we set the parameters to be more strongly risk-averse, then higher proportions will be allocated to lower-risk assets with maximum possible returns. Similarly, when we are more risk-tolerant, the algorithm will be willing to allocate more to assets with more risk, while seeking the best return on investment.

<img width="800" height="600" alt="risk_return_plot" src="https://github.com/user-attachments/assets/e99e19cc-4f17-40a7-aa24-2ce13e8bff87" />

