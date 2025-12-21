#!/usr/bin/env python3
"""
Version 4: rolling portfolio optimization with least squares and dynamic risk aversion via HMM regime detection (Viterbi training)
Inspired by https://stanford.edu/class/engr108/lectures/portfolio_slides.pdf
"""
import os, argparse, math, datetime as dt
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
try:
    from hmmlearn.hmm import GaussianHMM
    HAVE_HMM = True
except Exception:
    GaussianHMM = None
    HAVE_HMM = False

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

UNIVERSE = ["SPY", "QQQ", "IWM", "GOOGL", "EEM", "NVDA", "LQD", "MU", "MSFT", "TSM", "DIA"]
START = "2023-01-01"
END   = None
REBAL_FREQ = "B" # Daily
RETURN_LOOKBACK_DAYS = 100
EWMA_HALFLIFE_DAYS = 5
LAMBDA_RISK = 7.0 # Reduced base risk aversion for more aggressive allocation
GAMMA_TC = 0.001
TAU_TURNOVER = 0.40
W_MAX = 0.40 # Slightly relaxed max weight
HMM_LOOKBACK = 60 # Days for HMM features
REGIME_MULTIPLIERS = {0: 0.5, 1: 2.0, 2: 5.0} # More aggressive in bull (lower multiplier), conservative in bear
HMM_N_STATES = 3
HMM_RETRAIN_FREQ = 5  # days between HMM refits

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

    px = bars["close"].unstack(level=0)
    return px.sort_index()

def rebalance_alpaca_to_weights(target_w, notional):
    """Send market orders to reach target weights."""
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    trading_client = TradingClient(key, secret, paper=True)

    current_positions = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}

    for sym, target_w in target_w.items():
        target_notional = target_w * notional
        current_notional = current_positions.get(sym, 0.0)
        delta = target_notional - current_notional
        if abs(delta) < 1.0:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        rounded_notional = round(abs(delta), 2)
        if rounded_notional < 1.0:
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

# HMM helpers
def extract_hmm_features(returns_window):
    """Return standardized features array (n_days x n_features) for HMM: (cross-sectional mean, cross-sectional std)."""
    feats = np.column_stack([returns_window.mean(axis=1).values, returns_window.std(axis=1).values])
    scaler = StandardScaler()
    return scaler.fit_transform(feats)

def fit_hmm(features, n_states):
    """Fit a Gaussian HMM and return the trained model."""
    if not HAVE_HMM:
        raise RuntimeError("hmmlearn not available")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        algorithm='viterbi',  # Viterbi training: eliminates convergence warnings
        n_iter=10000,
        tol=1e-6,
        random_state=42
    )
    model.fit(features)
    if not model.monitor_.converged:
        raise RuntimeError("HMM did not converge")
    return model

def get_state_posteriors(model, X):
    """Return posterior probabilities for X (n_samples x n_states)."""
    try:
        return model.predict_proba(X)
    except Exception:
        means = model.means_
        dists = np.linalg.norm(means - X[0], axis=1)
        probs = np.exp(-dists)
        return (probs / probs.sum()).reshape(1, -1)

def compute_regime_multipliers(window_returns, state_sequence, n_states):
    """Map HMM states to regime multipliers using historical mean returns per state.
    Highest mean -> bull (low multiplier), lowest mean -> bear (high multiplier).
    """
    default = REGIME_MULTIPLIERS
    try:
        day_returns = window_returns.mean(axis=1)
        per_state_mean = np.full(n_states, np.nan)
        for s in range(n_states):
            mask = (state_sequence == s)
            vals = day_returns[mask]
            if len(vals) > 0:
                per_state_mean[s] = vals.mean()
        order = np.argsort(-np.nan_to_num(per_state_mean, nan=-np.inf))
        base = [0.5, 1.0, 3.0]  # Wider spread for stronger regime response
        multipliers = {}
        for rank, state in enumerate(order):
            multipliers[state] = base[rank] if rank < len(base) else base[-1]
        return multipliers
    except Exception:
        return default

def solve_portfolio(mu, Sigma, w_prev=None, max_invest_fraction=0.9, lambda_risk=LAMBDA_RISK):
    """
    Solve mean-variance portfolio with turnover regularization, using least-squares formulation.
    """
    n = len(mu)
    w = cp.Variable(n)
    
    L = np.linalg.cholesky(Sigma.values + 1e-8 * np.eye(n))  # Add small jitter for numerical stability
    
    obj = mu.values @ w - lambda_risk * cp.sum_squares(L.T @ w)
    
    constraints = [cp.sum(w) <= max_invest_fraction,
                   w >= 0,
                   w <= W_MAX]

    if w_prev is None:
        w_prev = np.zeros(n)
    turnover = cp.norm1(w - w_prev)
    obj = obj - GAMMA_TC * turnover
    constraints.append(turnover <= TAU_TURNOVER)

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, max_iter=10000)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Optimizer warning: {prob.status}")
    
    w_opt = pd.Series(np.clip(w.value if w.value is not None else np.zeros(n), 0, 1), index=mu.index)
    total_alloc = w_opt.sum()
    if total_alloc > max_invest_fraction:
        w_opt = w_opt * (max_invest_fraction / total_alloc)

    return w_opt

# Run Backtest
def walk_forward_backtest(px):
    global HAVE_HMM
    rets = px.pct_change().dropna()
    dates = rets.index
    rebal_dates = pd.date_range(dates[0], dates[-1], freq=REBAL_FREQ)
    rebal_dates = [d for d in rebal_dates if d in dates and d >= dates[RETURN_LOOKBACK_DAYS]]

    w_prev = None
    current_w = pd.Series(0, index=px.columns)
    equity = 1.0
    equity_curve = []
    weights_record = {}

    initial_window = rets.iloc[:HMM_LOOKBACK]
    feat_array = extract_hmm_features(initial_window)
    if HAVE_HMM:
        try:
            hmm = fit_hmm(feat_array, HMM_N_STATES)
            state_seq = hmm.predict(feat_array)
            regime_multipliers = compute_regime_multipliers(initial_window, state_seq, HMM_N_STATES)
        except Exception as e:
            print(f"Initial HMM failed: {e}")
            HAVE_HMM = False
            regime_multipliers = REGIME_MULTIPLIERS
    else:
        regime_multipliers = REGIME_MULTIPLIERS

    hm_available = HAVE_HMM

    for t_idx, today in enumerate(dates):
        if today in rebal_dates:
            window = rets.loc[:today].tail(RETURN_LOOKBACK_DAYS)
            cluster_window = window.tail(HMM_LOOKBACK)
            feat_array = extract_hmm_features(cluster_window)

            dynamic_lambda = LAMBDA_RISK
            if hm_available:
                should_refit = (t_idx % HMM_RETRAIN_FREQ == 0) or ('hmm' not in locals())
                if should_refit:
                    try:
                        hmm = fit_hmm(feat_array, HMM_N_STATES)
                        state_seq = hmm.predict(feat_array)
                        regime_multipliers = compute_regime_multipliers(cluster_window, state_seq, HMM_N_STATES)
                    except Exception:
                        hm_available = False

                if hm_available and 'hmm' in locals():
                    try:
                        probs = get_state_posteriors(hmm, feat_array[-1:].reshape(1, -1))[0]
                        dynamic_lambda = LAMBDA_RISK * sum(probs[i] * regime_multipliers.get(i, 1.0) for i in range(HMM_N_STATES))
                    except Exception:
                        dynamic_lambda = LAMBDA_RISK

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
        
        rets = px.pct_change().dropna()
        window = rets.tail(RETURN_LOOKBACK_DAYS)
        cluster_window = window.tail(HMM_LOOKBACK)
        feat_array = extract_hmm_features(cluster_window)
        
        dynamic_lambda = LAMBDA_RISK
        if HAVE_HMM:
            try:
                hmm = fit_hmm(feat_array, HMM_N_STATES)
                state_seq = hmm.predict(feat_array)
                regime_multipliers = compute_regime_multipliers(cluster_window, state_seq, HMM_N_STATES)
                probs = get_state_posteriors(hmm, feat_array[-1:].reshape(1, -1))[0]
                dynamic_lambda = LAMBDA_RISK * sum(probs[i] * regime_multipliers.get(i, 1.0) for i in range(HMM_N_STATES))
            except Exception:
                dynamic_lambda = LAMBDA_RISK

        mu = exp_weighted_mean_returns(window, EWMA_HALFLIFE_DAYS)
        Sigma = shrinkage_cov(window)
        w = solve_portfolio(mu, Sigma, w_prev=current_w.values, lambda_risk=dynamic_lambda)
        print("Target weights:\n", w.round(4))
        rebalance_alpaca_to_weights(w, notional=equity)

if __name__ == "__main__":
    main()