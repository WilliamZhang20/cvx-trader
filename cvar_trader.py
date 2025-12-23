import os, argparse, math, datetime as dt
import numpy as np
import pandas as pd
import cvxpy as cp

# Alpaca SDK
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# =====================
# CONFIG
# =====================
UNIVERSE = ["SPY", "QQQ", "TSLA", "GOOGL", "EEM", "NVDA", "MU", "MSFT", "AMZN"]
START = "2022-01-01"
END   = None

REBALANCE_FREQ_DAYS = 1
LOOKBACK_DAYS = 252

# Enhanced CVaR parameters
CONFIDENCE = 0.95 
LAMBDA_RET = 2.0
LAMBDA_CVAR = 5.0      
LAMBDA_MOMENTUM = 1.0
LAMBDA_TURNOVER = 0.001  
MAX_WEIGHT = 0.2
MAX_INVEST = 0.9
MIN_WEIGHT_THRESHOLD = 0.01

EXEC_BUFFER_ALPHA = 0.6
MIN_TRADE_PCT_NAV = 0.0025

# =====================
# Alpaca helpers
# =====================
def fetch_alpaca_prices(symbols, start, end):
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

def rebalance_alpaca_to_weights(target_w, current_w, notional):
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    trading_client = TradingClient(key, secret, paper=True)

    current_positions = {
        p.symbol: float(p.market_value)
        for p in trading_client.get_all_positions()
    }

    for sym, tgt_w in target_w.items():
        cur = current_positions.get(sym, 0.0)
        tgt = tgt_w * notional
        delta = tgt - cur

        if abs(delta) / notional < MIN_TRADE_PCT_NAV:
            continue

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        notional_trade = round(abs(delta), 2)
        if notional_trade < 1.0:
            continue

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=sym,
                notional=notional_trade,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )

def detect_regime(rets, lookback=63, crash_thresh=-0.08, cvar_thresh=0.02):
    """
    Returns: "risk_on", "neutral", "risk_off"

    Improvements:
    - Detects extreme single-day crashes (any asset loss below crash_thresh)
    - Computes a portfolio-level rolling CVaR and uses cvar_thresh as a fallback
    - Uses a z-score for recent vol versus historical cross-day vol
    """
    window = rets.tail(lookback)
    vol = window.std().mean()
    trend = window.mean().mean()

    # Use cross-day vol history to get a more stable z-score
    cross_day_vol = rets.std(axis=1).rolling(lookback).mean().dropna()
    if len(cross_day_vol) >= 2:
        vol_z = (vol - cross_day_vol.mean()) / (cross_day_vol.std() + 1e-9)
    else:
        vol_z = (vol - rets.std().mean()) / (rets.std().std() + 1e-9)

    # Extreme single-day loss across assets
    worst_single = window.min().min()

    # Portfolio-level CVaR (equal-weight proxy)
    port_rets = window.mean(axis=1)
    port_cvar = rolling_cvar(port_rets, alpha=CONFIDENCE)

    # Crash detection: highest priority
    if worst_single <= crash_thresh:
        # print(f"Detect_regime: {window.index[-1].date()} -> risk_off (reason=crash, worst={worst_single:.2%})")
        return "risk_off"

    # CVaR-based guard
    if port_cvar >= cvar_thresh:
        # print(f"Detect_regime: {window.index[-1].date()} -> risk_off (reason=cvar, port_cvar={port_cvar:.2%})")
        return "risk_off"

    # Volatility + negative trend
    if vol_z > 1.0 and trend < 0:
        # print(f"Detect_regime: {window.index[-1].date()} -> risk_off (reason=vol_trend, vol_z={vol_z:.2f}, trend={trend:.2%})")
        return "risk_off"
    elif vol_z < -0.5 and trend > 0:
        return "risk_on"
    else:
        return "neutral"

def get_rebalance_freq(regime):
    if regime == "risk_off":
        return 1      # daily
    elif regime == "neutral":
        return 5      # weekly
    else:
        return 10     # bi-weekly
    
def rolling_cvar(returns, alpha=0.95):
    losses = -returns
    var = np.quantile(losses, alpha)
    return losses[losses >= var].mean()

def get_cvar_limit(regime):
    if regime == "risk_off":
        return 0.015
    elif regime == "neutral":
        return 0.03
    else:
        return 0.05
    
def get_lambdas(regime):
    if regime == "risk_off":
        return dict(
            lambda_ret=0.5,
            lambda_cvar=10.0,
            lambda_momentum=0.0,
            lambda_turnover=0.005
        )
    elif regime == "neutral":
        return dict(
            lambda_ret=1.5,
            lambda_cvar=5.0,
            lambda_momentum=1.0,
            lambda_turnover=0.002
        )
    else:
        return dict(
            lambda_ret=3.0,
            lambda_cvar=2.0,
            lambda_momentum=2.0,
            lambda_turnover=0.001
        )

# =====================
# Enhanced CVaR Optimizer with Drawdown Control
# =====================
def cvar_term(losses, alpha, T):
    """
    Helper to construct CVaR constraint set
    Returns: cvar_expression, constraints
    
    Args:
        losses: cvxpy expression for losses
        alpha: confidence level
        T: number of samples
    """
    z = cp.Variable()
    u = cp.Variable(T)
    
    cvar_expr = z + (1 / ((1 - alpha) * T)) * cp.sum(u)
    constraints = [
        u >= losses - z,
        u >= 0,
        z >= 0
    ]

    return cvar_expr, constraints

def solve_enhanced_cvar_portfolio(
    returns: pd.DataFrame,
    w_prev=None,
    confidence=CONFIDENCE,
    lambda_ret=LAMBDA_RET,
    lambda_cvar=LAMBDA_CVAR,
    regime="neutral",
    lambda_momentum=LAMBDA_MOMENTUM,
    lambda_turnover=LAMBDA_TURNOVER,
):
    """
    Enhanced CVaR optimizer with:
    - Return CVaR (tail risk)
    - Volatility penalty removed for LP (set lambda_vol=0)
    - Momentum/trend following
    - Turnover penalty
    """
    R = returns.values
    T, N = R.shape
    
    # Decision variable
    w = cp.Variable(N)
    
    # Portfolio returns
    port_rets = R @ w
    
    # =========================
    # CVaR of losses (tail risk)
    # =========================
    losses = -port_rets
    cvar_ret, cvar_constraints = cvar_term(losses, confidence, T)

    # =========================
    # Expected return (with recency bias)
    # =========================
    decay = 0.94
    time_weights = np.array([decay ** (T - t - 1) for t in range(T)])
    time_weights = time_weights / time_weights.sum()
    
    weighted_mean = (R.T @ time_weights).reshape(-1)
    exp_ret = weighted_mean @ w
    
    # =========================
    # Momentum signal
    # =========================
    momentum_window = min(63, T // 4)
    recent_cum_ret = np.sum(R[-momentum_window:], axis=0)
    momentum_score = recent_cum_ret @ w
    
    # =========================
    # Turnover penalty
    # =========================
    if w_prev is not None:
        turnover = cp.norm1(w - w_prev)
    else:
        turnover = 0
    
    # =========================
    # Objective: Maximize return, minimize risks
    # =========================
    objective = cp.Maximize(
        lambda_ret * exp_ret
        + lambda_momentum * momentum_score
        - lambda_cvar * cvar_ret
        - lambda_turnover * turnover
    )
    
    # =========================
    # Constraints
    # =========================
    constraints = [
        cp.sum(w) == MAX_INVEST,
        w >= 0,
        w <= MAX_WEIGHT
    ]
    constraints += cvar_constraints
    
    # =========================
    # Solve
    # =========================
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(
            solver=cp.ECOS,
            verbose=False,
            max_iters=2000
        )
    except:
        # Fallback to SCS if ECOS fails
        try:
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                max_iters=2000,
                eps=1e-3
            )
        except:
            pass
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"Optimizer warning: {prob.status}")
    
    # Extract and clean weights
    w_opt = np.zeros(N) if w.value is None else w.value
    w_opt = np.clip(w_opt, 0, MAX_WEIGHT)
    
    # Threshold small weights
    w_opt[w_opt < MIN_WEIGHT_THRESHOLD] = 0.0
    
    # No renormalization to allow cash holdings
    
    return pd.Series(w_opt, index=returns.columns)

# =====================
# Backtest
# =====================
def walk_forward_backtest(px):
    rets = px.pct_change().dropna()
    dates = rets.index
    
    # Rebalance schedule
    last_rebalance = -1
    prev_regime = None
    
    w = pd.Series(0.0, index=px.columns)
    equity = 1.0
    curve = []
    weights_record = {}
    
    for t, day in enumerate(dates):
        # Rebalance if it's a rebalance day
        if t >= LOOKBACK_DAYS:
            window = rets.iloc[t - LOOKBACK_DAYS:t]
            regime = detect_regime(window)
            if regime != prev_regime:
                print(f"Regime change on {day.date()}: {prev_regime} -> {regime}")
                prev_regime = regime
            freq = get_rebalance_freq(regime)
            if (t - last_rebalance) >= freq:
                lambdas = get_lambdas(regime)
            
            try:
                w_new = solve_enhanced_cvar_portfolio(
                    window,
                    w_prev=w.values,
                    regime=regime,
                    **lambdas,
                )
                
                # Execution buffer to reduce turnover
                alpha = 0.5 if regime=="risk_off" else 0.9
                w = alpha * w_new + (1-alpha) * w

                last_rebalance = t
                
                # Cap if sum > MAX_INVEST (unlikely but safe)
                if w.sum() > MAX_INVEST:
                    w *= MAX_INVEST / w.sum()
                
                weights_record[day] = w.copy()
            except Exception as e:
                print(f"Optimization failed at {day}: {e}")
        
        # Apply daily returns
        if t > 0 and w.sum() > 0:
            equity *= 1.0 + float(rets.iloc[t] @ w)

        if equity < 0.92 * max([e for _, e in curve], default=equity):
            w *= 0.5
        
        curve.append((day, equity))
    
    curve = pd.Series(dict(curve))
    weights = pd.DataFrame(weights_record).T.reindex(curve.index).ffill().fillna(0.0)
    return curve, weights

# =====================
# Performance metrics
# =====================
def calculate_metrics(curve):
    daily = curve.pct_change().dropna()
    
    # Annualized metrics
    n_days = len(curve)
    ann_ret = (curve.iloc[-1] / curve.iloc[0]) ** (252 / n_days) - 1
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    
    # Max drawdown
    cummax = curve.cummax()
    drawdown = (curve - cummax) / cummax
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    
    return {
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar
    }

# =====================
# CLI
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    px = fetch_alpaca_prices(UNIVERSE, START, END)

    if args.backtest:
        print("Running backtest with enhanced CVaR + Drawdown control...")
        curve, weights = walk_forward_backtest(px)
        
        metrics = calculate_metrics(curve)
        print(f"\n=== Backtest Results ===")
        print(f"Return:     {metrics['ann_return']:>7.2%}")
        print(f"Volatility: {metrics['ann_vol']:>7.2%}")
        print(f"Sharpe:     {metrics['sharpe']:>7.2f}")
        print(f"Max DD:     {metrics['max_dd']:>7.2%}")
        print(f"Calmar:     {metrics['calmar']:>7.2f}")
        
        print(f"\nFinal weights:\n{weights.iloc[-1].round(4)}")

    if args.paper:
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
        trading_client = TradingClient(key, secret, paper=True)
        equity = float(trading_client.get_account().equity)

        current_positions = {
            p.symbol: float(p.market_value)
            for p in trading_client.get_all_positions()
            if p.symbol in UNIVERSE
        }

        current_w = pd.Series(
            {s: current_positions.get(s, 0.0) / equity for s in UNIVERSE}
        )

        rets = px.pct_change().dropna()
        window = rets.tail(LOOKBACK_DAYS)

        print("Optimizing portfolio with enhanced CVaR strategy...")
        regime = detect_regime(window)
        print(f"Paper trading regime on {window.index[-1].date()}: {regime}")
        lambdas = get_lambdas(regime)
        target_w = solve_enhanced_cvar_portfolio(window, w_prev=current_w.values, regime=regime, **lambdas)
        target_w = EXEC_BUFFER_ALPHA * target_w + (1 - EXEC_BUFFER_ALPHA) * current_w
        # No normalization to allow cash

        print("\n=== Target Weights ===")
        print(target_w.round(4))
        
        print("\n=== Current Weights ===")
        print(current_w.round(4))
        
        print("\nExecuting trades...")
        rebalance_alpaca_to_weights(target_w, current_w, equity)
        print("Done!")

if __name__ == "__main__":
    main()