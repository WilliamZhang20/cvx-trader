## Versions contained in this folder

Version 1: simple convex optimization with quadratic programming and data smoothing with EWMA and Ledoit-Wolf covariance matrix. 

Version 2: more robust convex least-squares objective rather than quadratic form, improved numerical stabilty with Cholesky Factorization. After some parameter tuning, running backtest from 2023/01/01 to September 19, 2025 yields an annualized return of 34.35%, volatility of 12.34%, and a nice Sharpe ratio of 2.78.

Version 3: new logic to detect market shifts for dynamic risk aversion using k-means clustering