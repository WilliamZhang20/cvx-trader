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
UNIVERSE = ["SPY", "ITA", "IWM", "GOOGL", "EEM", "NVDA", "MU", "MSFT", "QQQ"]
START = "2022-01-01"
END   = None

REBALANCE_FREQ_DAYS = 21  # Monthly rebalance
LOOKBACK_DAYS = 252       # 1 year (reduced from 2 for speed)

# Enhanced CVaR parameters
CONFIDENCE = 0.95
LAMBDA_RET = 3.0           # Return reward (increased from 1.0)
LAMBDA_CVAR = 3.0          # Tail risk penalty (reduced from 5.0)
LAMBDA_VOL = 1.0           # Volatility penalty (reduced from 2.0)
LAMBDA_MOMENTUM = 1.5      # Momentum/trend bonus
LAMBDA_TURNOVER = 0.015    # Transaction cost proxy (reduced)

MAX_WEIGHT = 0.30          # Increased from 0.25
MAX_INVEST = 1.0           # Increased from 0.95 (be fully invested)
MIN_WEIGHT_THRESHOLD = 0.01

EXEC_BUFFER_ALPHA = 0.9
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
        T: number of samples (must be passed explicitly)
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
    lambda_vol=LAMBDA_VOL,
    lambda_momentum=LAMBDA_MOMENTUM,
    lambda_turnover=LAMBDA_TURNOVER,
):
    """
    Enhanced CVaR optimizer with:
    - Return CVaR (tail risk)
    - Volatility penalty (faster than drawdown CVaR)
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
    # 1. CVaR of losses (tail risk)
    # =========================
    losses = -port_rets
    cvar_ret, cvar_constraints = cvar_term(losses, confidence, T)
    
    # =========================
    # 2. Volatility (much faster than drawdown CVaR)
    # =========================
    # Compute covariance matrix
    cov = np.cov(R.T)
    portfolio_variance = cp.quad_form(w, cov)
    
    # =========================
    # 3. Expected return (with recency bias)
    # =========================
    # Weight recent returns more heavily (exponential decay)
    decay = 0.94
    time_weights = np.array([decay ** (T - t - 1) for t in range(T)])
    time_weights = time_weights / time_weights.sum()
    
    weighted_mean = (R.T @ time_weights).reshape(-1)
    exp_ret = weighted_mean @ w
    
    # =========================
    # 4. Momentum signal
    # =========================
    # Assets with positive recent trend get a bonus
    # Use last 63 days (3 months) for momentum
    momentum_window = min(63, T // 4)
    recent_cum_ret = np.sum(R[-momentum_window:], axis=0)
    momentum_score = recent_cum_ret @ w
    
    # =========================
    # 5. Turnover penalty
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
        - lambda_vol * portfolio_variance
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
    
    # Renormalize
    if w_opt.sum() > 0:
        w_opt *= MAX_INVEST / w_opt.sum()
    
    return pd.Series(w_opt, index=returns.columns)

# =====================
# Backtest
# =====================
def walk_forward_backtest(px):
    rets = px.pct_change().dropna()
    dates = rets.index
    
    # Rebalance schedule
    rebal_indices = list(range(LOOKBACK_DAYS, len(dates), REBALANCE_FREQ_DAYS))
    
    w = pd.Series(0.0, index=px.columns)
    equity = 1.0
    curve = []
    weights_record = {}
    
    for t, day in enumerate(dates):
        # Rebalance if it's a rebalance day
        if t in rebal_indices:
            window = rets.iloc[t - LOOKBACK_DAYS:t]
            
            try:
                w_new = solve_enhanced_cvar_portfolio(window, w_prev=w.values)
                
                # Execution buffer to reduce turnover
                w = EXEC_BUFFER_ALPHA * w_new + (1 - EXEC_BUFFER_ALPHA) * w
                w /= w.sum() if w.sum() > 0 else 1.0
                
                weights_record[day] = w.copy()
            except Exception as e:
                print(f"Optimization failed at {day}: {e}")
        
        # Apply daily returns
        if t > 0 and w.sum() > 0:
            equity *= 1.0 + float(rets.iloc[t] @ w)
        
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
        target_w = solve_enhanced_cvar_portfolio(window, w_prev=current_w.values)
        target_w = EXEC_BUFFER_ALPHA * target_w + (1 - EXEC_BUFFER_ALPHA) * current_w
        target_w /= target_w.sum()

        print("\n=== Target Weights ===")
        print(target_w.round(4))
        
        print("\n=== Current Weights ===")
        print(current_w.round(4))
        
        print("\nExecuting trades...")
        rebalance_alpaca_to_weights(target_w, current_w, equity)
        print("Done!")

if __name__ == "__main__":
    main()