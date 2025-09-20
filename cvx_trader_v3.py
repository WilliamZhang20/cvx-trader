#!/usr/bin/env python3
"""
Version 3: rolling portfolio optimization with least squares and dynamic risk aversion via k-means clustering
Inspired by https://stanford.edu/class/engr108/lectures/portfolio_slides.pdf
"""
import os, argparse, math, datetime as dt
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "MSFT", "TSM", "DIA"]
START = "2023-01-01"
END   = None
REBAL_FREQ = "B" # Daily
RETURN_LOOKBACK_DAYS = 252
EWMA_HALFLIFE_DAYS = 10
LAMBDA_RISK = 3.0 # Base risk aversion
GAMMA_TC = 0.001
TAU_TURNOVER = 0.40
W_MAX = 0.35
CLUSTER_LOOKBACK = 60 # Days for clustering features
REGIME_MULTIPLIERS = {0: 0.3, 1: 1.0, 2: 2} # Bull, Neutral, Bear (less aggressive)

# -----------------------
# Alpaca helpers
# -----------------------
def fetch_alpaca_prices(symbols, start, end):
    """Fetch daily OHLCV bars from Alpaca Data API."""
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    client = StockHistoricalDataClient(key, secret)

    if end is None:
        end = dt.date.today()

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
    )
    bars = client.get_stock_bars(request).df

    # Alpaca returns multi-index (symbol, timestamp)
    px = bars["close"].unstack(level=0)
    return px.sort_index()

def rebalance_alpaca_to_weights(target_w, notional):
    """Send market orders to reach target weights."""
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    trading_client = TradingClient(key, secret, paper=True)

    # Fetch current positions
    current_positions = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}

    # Submit new notional orders 
    for sym, target_w in target_w.items():
        target_notional = target_w * notional
        current_notional = current_positions.get(sym, 0.0)
        delta = target_notional - current_notional
        if abs(delta) < 1.0:  # skip tiny adjustments
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        # Round notional value to 2 decimal places
        rounded_notional = round(abs(delta), 2)
        if rounded_notional < 1.0:  # Ensure notional is at least $1 after rounding
            continue
        req = MarketOrderRequest(
            symbol=sym,
            notional=rounded_notional,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(req)

# -----------------------
# Optimizer
# -----------------------
def exp_weighted_mean_returns(returns, halflife_days):
    lam = math.log(2)/halflife_days
    w = np.exp(-lam * np.arange(len(returns))[::-1])
    w = w / w.sum()
    mu = (returns * w[:,None]).sum(axis=0)
    return pd.Series(mu, index=returns.columns)

def shrinkage_cov(returns):
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def solve_portfolio(mu, Sigma, w_prev=None, max_invest_fraction=0.8, lambda_risk=LAMBDA_RISK):
    """
    Solve mean-variance portfolio with turnover regularization, using least-squares formulation.
    
    max_invest_fraction: fraction of total capital to invest (rest is cash)
    lambda_risk: dynamic risk aversion parameter
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Cholesky decomposition for least-squares form
    L = np.linalg.cholesky(Sigma.values)  # Sigma = L @ L.T
    
    obj = mu.values @ w - lambda_risk * cp.sum_squares(L.T @ w)
    
    constraints = [cp.sum(w) <= max_invest_fraction,  # only invest up to max fraction
                   w >= 0, # no shorting
                   w <= W_MAX]

    # Turnover regularization
    if w_prev is None:
        w_prev = np.zeros(n)
    turnover = cp.norm1(w - w_prev)
    obj = obj - GAMMA_TC * turnover
    constraints.append(turnover <= TAU_TURNOVER)

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    w_opt = pd.Series(np.clip(w.value, 0, 1), index=mu.index)
    # Normalize to sum to at most max_invest_fraction
    total_alloc = w_opt.sum()
    if total_alloc > max_invest_fraction:
        w_opt = w_opt * (max_invest_fraction / total_alloc)

    return w_opt

# Run Backtest to determine objective parameters
def walk_forward_backtest(px):
    rets = px.pct_change().dropna()
    dates = rets.index
    rebal_dates = pd.date_range(dates[0], dates[-1], freq=REBAL_FREQ)
    rebal_dates = [d for d in rebal_dates if d in dates and d >= dates[RETURN_LOOKBACK_DAYS]]

    w_prev = None
    current_w = pd.Series(0, index=px.columns)
    equity = 1.0
    equity_curve = []
    weights_record = {}

    # Pre-fit k-means on initial window
    initial_window = rets.iloc[:CLUSTER_LOOKBACK]
    features = np.column_stack([initial_window.mean(axis=1), initial_window.std(axis=1)])
    kmeans = KMeans(n_clusters=3, random_state=42).fit(features)

    for t_idx, today in enumerate(dates):
        if today in rebal_dates:
            window = rets.loc[:today].tail(RETURN_LOOKBACK_DAYS)
            # Compute clustering features
            cluster_window = window.tail(CLUSTER_LOOKBACK)
            features = np.column_stack([cluster_window.mean(axis=1), cluster_window.std(axis=1)])
            
            # Update k-means more frequently
            if t_idx % 10 == 0:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
            
            # Use soft clustering (probability-weighted lambda)
            current_features = np.array([[window.iloc[-1].mean(), window.iloc[-1].std()]])
            # Compute distances to centroids and convert to probabilities
            distances = kmeans.transform(current_features)[0]
            probs = np.exp(-distances) / np.exp(-distances).sum()  # Softmax-like
            dynamic_lambda = LAMBDA_RISK * sum(probs[i] * REGIME_MULTIPLIERS[i] for i in range(3))

            mu = exp_weighted_mean_returns(window, EWMA_HALFLIFE_DAYS)
            Sigma = shrinkage_cov(window)
            current_w = solve_portfolio(mu, Sigma, w_prev, lambda_risk=dynamic_lambda)
            weights_record[today] = current_w
            w_prev = current_w.values.copy()

        if t_idx > 0:
            day_ret = float((rets.loc[today] * current_w).sum())
            equity *= (1.0 + day_ret)
        equity_curve.append((today, equity))

    curve = pd.Series(dict(equity_curve)).sort_index().rename("Equity")
    weights_panel = pd.DataFrame(weights_record).T.reindex(curve.index).ffill().fillna(0.0)
    return curve, weights_panel

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--notional", type=float, default=10000.0)
    args = parser.parse_args()

    px = fetch_alpaca_prices(UNIVERSE, start=START, end=END)

    if args.backtest:
        curve, weights = walk_forward_backtest(px)
        ann_ret = (curve.iloc[-1] / curve.iloc[0]) ** (252/len(curve)) - 1
        daily_rets = curve.pct_change().dropna()
        ann_vol = daily_rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        print(f"Backtest: Return {ann_ret:.2%}, Volatility {ann_vol:.2%}, Sharpe {sharpe:.2f}")

    if args.paper:
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
        trading_client = TradingClient(key, secret, paper=True)
        account = trading_client.get_account()
        equity = float(account.equity)
        current_positions = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions() if p.symbol in UNIVERSE}
        current_w = pd.Series({sym: current_positions.get(sym, 0.0) / equity for sym in UNIVERSE}) if equity > 0 else pd.Series(0.0, index=UNIVERSE)
        
        # Build today's allocation with soft clustering
        rets = px.pct_change().dropna()
        window = rets.tail(RETURN_LOOKBACK_DAYS)
        cluster_window = window.tail(CLUSTER_LOOKBACK)
        features = np.column_stack([cluster_window.mean(axis=1), cluster_window.std(axis=1)])
        kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
        current_features = np.array([[window.iloc[-1].mean(), window.iloc[-1].std()]])
        distances = kmeans.transform(current_features)[0]
        probs = np.exp(-distances) / np.exp(-distances).sum()
        dynamic_lambda = LAMBDA_RISK * sum(probs[i] * REGIME_MULTIPLIERS[i] for i in range(3))
        
        mu = exp_weighted_mean_returns(window, EWMA_HALFLIFE_DAYS)
        Sigma = shrinkage_cov(window)
        w = solve_portfolio(mu, Sigma, w_prev=current_w.values, lambda_risk=dynamic_lambda)
        print("Target weights:\n", w.round(4))
        rebalance_alpaca_to_weights(w, notional=equity)

if __name__ == "__main__":
    main()