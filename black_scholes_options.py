#!/usr/bin/env python3
import time
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
import os

# Alpaca trading client - used for equity hedges and (example) option orders
# install: pip install alpaca-trade-api
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import OptionOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    # Fallback for older SDKs or if not installed.
    TradingClient = None
    OptionOrderRequest = None
    OrderSide = None
    TimeInForce = None

# ---------- CONFIG ----------
# API keys are read when needed via get_alpaca_keys() (uses os.environ[...] and will raise if missing)
API_KEY = None
API_SECRET = None
APCA_PAPER = True


def get_alpaca_keys():
    """Return (key, secret) from environment using os.environ[...] so missing keys raise clearly."""
    try:
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
    except KeyError as e:
        raise RuntimeError(f"Missing Alpaca env var {e.args[0]}. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your environment.")
    return key, secret

# Universe & filters
UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]  # keep small for paper
DTE_MIN = 14
DTE_MAX = 45
MIN_OI = 200
MIN_VOL = 50
DELTA_BAND = (0.25, 0.60)  # absolute delta band
MAX_POSITION_NOTIONAL = 10000.0  # per-trade notional cap in USD
MAX_NET_DELTA = 5.0  # shares-equivalent net delta cap
REPRICE_INTERVAL = 1.0  # seconds between main loops
TOP_K = 2  # number of candidate contracts to attempt per loop

RISK_FREE_RATE = 0.02  # annualized risk-free rate used if needed

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alpaca_bs_trader")

# ---------- DATA STRUCTS ----------
@dataclass
class OptionQuote:
    symbol: str           # unique option symbol like AAPL240119C00100000
    underlying: str
    strike: float
    expiry: pd.Timestamp
    option_type: str      # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    delta: float = None   # if feed provides it
    iv: float = None      # implied volatility if provided
    # convenience:
    @property
    def mid(self):
        if self.bid is not None and self.ask is not None and (self.bid + self.ask) > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

# ---------- BLACK-SCHOLES IMPLEMENTATION ----------
def bs_d1(S, K, r, sigma, T):
    # handle degenerate sigma or T
    if sigma <= 0 or T <= 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_price_and_greeks_vectorized(S_arr, K_arr, r_arr, sigma_arr, T_arr, opt_type_arr):
    """
    Vectorized Black-Scholes for arrays (NumPy).
    Returns dict with price, delta, gamma, vega, theta (per-contract arrays).
    opt_type_arr: array of 'call' or 'put'
    """
    S = np.asarray(S_arr, dtype=float)
    K = np.asarray(K_arr, dtype=float)
    sigma = np.asarray(sigma_arr, dtype=float)
    T = np.asarray(T_arr, dtype=float)
    r = np.asarray(r_arr, dtype=float)

    eps = 1e-12
    sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(np.maximum(S, eps) / np.maximum(K, eps)) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT + 1e-12)
    d2 = d1 - sigma * sqrtT

    nd1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    call_price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    price = np.where(np.asarray(opt_type_arr) == "call", call_price, put_price)

    # Delta
    delta_call = cdf_d1
    delta_put = cdf_d1 - 1.0
    delta = np.where(np.asarray(opt_type_arr) == "call", delta_call, delta_put)

    gamma = nd1 / (S * sigma * sqrtT + 1e-12)
    vega = S * nd1 * sqrtT
    # Theta (per-year) approximate
    theta_call = (-S * nd1 * sigma) / (2 * sqrtT) - r * K * np.exp(-r * T) * cdf_d2
    theta_put = (-S * nd1 * sigma) / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = np.where(np.asarray(opt_type_arr) == "call", theta_call, theta_put)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta
    }

# ---------- HELPERS ----------
def days_to_expiry(expiry_timestamp: pd.Timestamp, now_ts: pd.Timestamp) -> float:
    # returns fraction of years (365)
    delta = expiry_timestamp - now_ts
    return max(delta.total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)

def historical_volatility_from_series(close_series: pd.Series, annualize=252):
    # log returns
    rets = np.log(close_series).diff().dropna()
    if len(rets) < 2:
        return 0.2
    return rets.std() * math.sqrt(annualize)

# Option Chain fetch
# - Alpaca Options API (if you have access) -> return OptionQuote items
def fetch_option_chain(underlying: str, now_ts: pd.Timestamp = None, S_override: float = None) -> List[OptionQuote]:
    """Return a list of OptionQuote for the underlying.

    Tries to use an exchange/broker feed when available; otherwise returns a synthetic chain generated
    from the Black-Scholes model with reasonable spreads, vols, and liquidity for backtesting/simulation.
    """
    now_ts = now_ts or pd.Timestamp.utcnow()

    # Try to get a live price
    S = S_override
    if S is None:
        try:
            client = init_alpaca_client()
            last_trade = client.get_latest_trade(underlying)
            S = float(last_trade.price)
        except Exception:
            try:
                import yfinance as yf
                hist = yf.Ticker(underlying).history(period="120d")['Close'].dropna()
                if len(hist) > 0:
                    S = float(hist.iloc[-1])
            except Exception:
                S = 100.0

    # estimate historical vol
    hist_sigma = 0.25
    try:
        import yfinance as yf
        hist = yf.Ticker(underlying).history(period="120d")['Close'].dropna()
        if len(hist) >= 10:
            hist_sigma = historical_volatility_from_series(hist)
    except Exception:
        pass

    strikes = list(np.linspace(max(1.0, S * 0.5), S * 1.5, num=21))
    # pick a few expiries in DTE range
    expiries = []
    dte_candidates = np.unique([DTE_MIN, (DTE_MIN + DTE_MAX) // 2, DTE_MAX])
    for d in dte_candidates:
        expiries.append((now_ts + pd.Timedelta(days=int(d))).normalize())

    quotes = []
    rng = np.random.default_rng(42)
    for expiry in expiries:
        T = days_to_expiry(expiry, now_ts)
        if T <= 0:
            continue
        Ks = strikes
        opt_types = ['call'] * len(Ks) + ['put'] * len(Ks)
        K_arr = np.array(Ks + Ks)
        S_arr = np.full(len(K_arr), S)
        r_arr = np.full(len(K_arr), RISK_FREE_RATE)
        sigma_arr = np.full(len(K_arr), hist_sigma)
        Ts = np.full(len(K_arr), T)
        res = bs_price_and_greeks_vectorized(S_arr, K_arr, r_arr, sigma_arr, Ts, opt_types)
        prices = res['price']
        deltas = res['delta']

        for i in range(len(K_arr)):
            k = float(K_arr[i])
            opt_type = opt_types[i]
            bs_mid = float(prices[i])
            # inject realistic relative spread and some noise
            rel_spread = max(0.01, 0.03 * rng.random())
            mid = max(0.001, bs_mid * (1.0 + rng.normal(0, 0.01)))
            spread = mid * rel_spread
            bid = max(0.0, mid - spread / 2.0)
            ask = mid + spread / 2.0
            last = mid
            vol = int(rng.integers(50, 500))
            oi = int(rng.integers(50, 2000))
            quote = OptionQuote(
                symbol=f"{underlying}_{expiry.date().isoformat()}_{opt_type[0].upper()}_{int(k)}",
                underlying=underlying,
                strike=k,
                expiry=expiry,
                option_type=opt_type,
                bid=bid,
                ask=ask,
                last=last,
                volume=vol,
                open_interest=oi,
                delta=float(deltas[i]),
                iv=float(hist_sigma)
            )
            quotes.append(quote)

    return quotes

def init_alpaca_client():
    if TradingClient is None:
        raise RuntimeError("Alpaca TradingClient not available. Please install alpaca-trade-api or use official SDK.")
    key, secret = get_alpaca_keys()
    client = TradingClient(key, secret, paper=APCA_PAPER)
    return client

def submit_option_order_example(client, symbol: str, qty: int, side: str):
    """
    Example submission using Alpaca OptionOrderRequest (adjust depending on your SDK version).
    This function demonstrates how to place option orders; adapt to your environment.
    """
    if OptionOrderRequest is None:
        raise RuntimeError("Option order class not available in current SDK import. Adjust to your SDK version.")

    # Example: market order, day time in force
    order = OptionOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        type="market",
        time_in_force=TimeInForce.DAY
    )
    r = client.submit_order(order)
    return r

def submit_equity_order(client, symbol: str, qty: int, side: str):
    # Simple market order for hedge (equity)
    logger.info("Submitting equity hedge order: %s %d %s", symbol, qty, side)
    try:
        return client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
    except Exception as e:
        logger.exception("Equity order failed: %s", e)
        return None

# ---------- SIMPLE RISK CHECKS ----------
def within_risk_limits(portfolio_state: Dict[str, Any], option_quote: OptionQuote, direction: str) -> bool:
    # crude: check position notional and net delta.
    # portfolio_state contains {'net_delta_shares': float, 'notional_open': float}
    # direction: 'buy' or 'sell' (from perspective of buying option)
    expected_notional = abs(option_quote.mid) * 100.0  # one contract controls 100 shares
    if portfolio_state.get("notional_open", 0.0) + expected_notional > MAX_POSITION_NOTIONAL:
        logger.warning("Notional limit hit, skipping trade.")
        return False
    if abs(portfolio_state.get("net_delta_shares", 0.0)) > MAX_NET_DELTA:
        logger.warning("Net delta limit exceeded, skipping trade.")
        return False
    return True

# ---------- MAIN LOOP ----------
def main_loop():
    # init client (for equity hedges)
    try:
        client = init_alpaca_client()
        logger.info("Alpaca client initialized.")
    except Exception as e:
        logger.exception("Couldn't init Alpaca client: %s", e)
        client = None

    # very simple portfolio state
    portfolio_state = {"net_delta_shares": 0.0, "notional_open": 0.0}

    last_run = 0.0
    while True:
        now_ts = pd.Timestamp.utcnow()
        t0 = time.time()
        try:
            candidates = []

            # Loop the universe (small)
            for sym in UNIVERSE:
                # 1) fetch underlying price history & last quote
                # For simplicity we use Alpaca if available
                try:
                    if client is not None:
                        # Using TradingClient: get last trade (this may depend on SDK)
                        # This code may need changing per alpaca SDK version.
                        last_trade = client.get_latest_trade(sym)
                        S = float(last_trade.price)
                    else:
                        # fallback placeholder price
                        S = 100.0
                except Exception:
                    # if live fetch fails, skip symbol
                    logger.exception("Failed to fetch last trade for %s", sym)
                    continue

                # 2) fetch option chain for this underlying (user must implement)
                chain = fetch_option_chain(sym)
                if not chain:
                    logger.debug("No chain returned for %s — ensure fetch_option_chain implemented", sym)
                    continue

                # 3) filter chain by expiry DTE and liquidity
                filtered = []
                for q in chain:
                    T = days_to_expiry(q.expiry, now_ts)
                    dte_days = (q.expiry - now_ts).days
                    if dte_days < DTE_MIN or dte_days > DTE_MAX:
                        continue
                    if q.open_interest < MIN_OI or q.volume < MIN_VOL:
                        continue
                    if (q.bid is None or q.ask is None) or (q.ask - q.bid) / max(q.mid, 0.001) > 0.10:
                        # too wide relative spread
                        continue
                    filtered.append(q)

                if not filtered:
                    continue

                # 4) build arrays for vectorized computation
                Ks = [q.strike for q in filtered]
                Ts = [days_to_expiry(q.expiry, now_ts) for q in filtered]
                opt_types = [q.option_type for q in filtered]
                bids = np.array([q.bid for q in filtered], dtype=float)
                asks = np.array([q.ask for q in filtered], dtype=float)
                mids = (bids + asks) / 2.0

                # 5) volatility: use feed IV if available, else fallback to historical vol
                ivs = []
                # get historical vol quick fallback: short 60-day window from Alpaca bars (if available)
                hist_sigma = 0.25
                try:
                    if client is not None:
                        barset = client.get_bars(sym, timeframe="1Day", start=(now_ts - pd.Timedelta(days=90)).isoformat(), end=now_ts.isoformat(), limit=90)
                        # adapt depending on returned object shape; attempt to extract close prices
                        closes = []
                        for bar in barset:
                            # many SDKs return objects with .c or .close
                            c = getattr(bar, "c", None) or getattr(bar, "close", None)
                            if c is not None:
                                closes.append(float(c))
                        if len(closes) >= 10:
                            hist_sigma = historical_volatility_from_series(pd.Series(closes))
                except Exception:
                    logger.debug("Couldn't fetch historical bars for %s; using default sigma", sym)

                for q in filtered:
                    if q.iv is not None and q.iv > 0:
                        ivs.append(q.iv)
                    else:
                        ivs.append(hist_sigma)

                r_arr = np.full(len(Ks), RISK_FREE_RATE)
                S_arr = np.full(len(Ks), S)
                # 6) compute BS prices & greeks vectorized
                results = bs_price_and_greeks_vectorized(S_arr, Ks, r_arr, ivs, Ts, opt_types)

                bm_price = results["price"]
                deltas = results["delta"]
                vegas = results["vega"]
                # 7) score mispricing: (market mid - model) normalized by spread
                spreads = asks - bids
                spreads = np.where(spreads <= 0, 1e-6, spreads)
                mispricing = mids - bm_price
                score = mispricing / spreads

                # 8) collect candidates meeting delta band
                for i, q in enumerate(filtered):
                    delta_abs = abs(deltas[i])
                    if delta_abs < DELTA_BAND[0] or delta_abs > DELTA_BAND[1]:
                        continue
                    candidates.append({
                        "underlying": sym,
                        "quote": q,
                        "score": float(score[i]),
                        "mid": float(mids[i]),
                        "bs_price": float(bm_price[i]),
                        "delta": float(deltas[i]),
                        "vega": float(vegas[i]),
                        "spread": float(spreads[i])
                    })

            # 9) pick top-K by absolute score
            if not candidates:
                logger.debug("No candidates found this cycle.")
            else:
                cands_df = pd.DataFrame(candidates)
                cands_df["abs_score"] = cands_df["score"].abs()
                cands_df = cands_df.sort_values("abs_score", ascending=False).head(TOP_K)
                logger.info("Top candidates:\n%s", cands_df[["underlying", "quote", "score", "mid", "bs_price", "delta"]])

                # 10) attempt execution for each candidate (simple strategy: buy if market < bs_price - thresh, sell if market > bs_price + thresh)
                for idx, row in cands_df.iterrows():
                    q: OptionQuote = row["quote"]
                    score = row["score"]
                    mid = row["mid"]
                    bs_price = row["bs_price"]
                    delta = row["delta"]
                    # threshold to avoid tiny edges
                    threshold_ticks = max(0.01, row["spread"] * 0.5)
                    # decide side
                    if bs_price - mid > threshold_ticks:
                        # market is cheaper than model -> BUY option (expect reversion up)
                        side = "buy"
                    elif mid - bs_price > threshold_ticks:
                        # market is dearer -> SELL option
                        side = "sell"
                    else:
                        continue

                    # Risk check
                    if not within_risk_limits(portfolio_state, q, side):
                        continue

                    # size: crude: 1 contract unless notional cap
                    qty = 1
                    notional = abs(q.mid) * 100.0 * qty
                    if portfolio_state["notional_open"] + notional > MAX_POSITION_NOTIONAL:
                        logger.info("Would exceed notional cap; reduce qty or skip")
                        continue

                    # Submit option order (example) - you must implement according to your SDK
                    try:
                        if client is not None and OptionOrderRequest is not None:
                            logger.info("Submitting option %s %s %d @ mid %.4f", q.symbol, side, qty, q.mid)
                            # Example call - adjust for your SDK; this may not match your alpaca sdk exactly.
                            resp = submit_option_order_example(client, q.symbol, qty, side)
                            logger.info("Option order response: %s", resp)
                        else:
                            logger.info("Would %s option %s qty %d (paper run)", side, q.symbol, qty)
                            resp = None
                    except Exception:
                        logger.exception("Option order submission failed for %s", q.symbol)
                        resp = None

                    # Update portfolio state: naive update (one contract = 100 shares delta exposure)
                    # For buy: add delta exposure; for sell: subtract
                    sign = 1.0 if side == "buy" else -1.0
                    portfolio_state["net_delta_shares"] += sign * delta * 100.0 * qty
                    portfolio_state["notional_open"] += notional

                    # Hedge to neutralize delta (simple immediate hedge)
                    # Compute hedge qty in shares to bring net_delta_shares to zero (rounded to nearest share)
                    if abs(portfolio_state["net_delta_shares"]) > 1.0:  # only hedge if > 1 share exposure
                        hedge_shares = int(round(-portfolio_state["net_delta_shares"]))
                        hedge_side = "buy" if hedge_shares > 0 else "sell"
                        hedge_qty = abs(hedge_shares)
                        if client is not None:
                            try:
                                r = submit_equity_order(client, q.underlying, hedge_qty, hedge_side)
                                logger.info("Hedge order submitted: %s", r)
                                # assume fully filled (paper); update net_delta_shares
                                portfolio_state["net_delta_shares"] += hedge_shares
                            except Exception:
                                logger.exception("Hedge failed")
                        else:
                            logger.info("Would hedge %d shares %s", hedge_qty, hedge_side)
                            portfolio_state["net_delta_shares"] += hedge_shares

            # end main candidate handling

        except Exception:
            logger.exception("Main loop exception")

        # sleep until next iteration
        elapsed = time.time() - t0
        to_sleep = max(0.01, REPRICE_INTERVAL - elapsed)
        time.sleep(to_sleep)


def simulate_backtest(initial_capital: float = 100000.0, tc_rate: float = 0.001, slippage_pct: float = 0.0005):
    """Simple backtest over historical underlying prices using the scanning + one-day hold logic.

    - Uses yfinance (fallback) to obtain price series.
    - For each day t (except last), we scan the synthetic chain at t, pick candidates, execute at mid, hedge with shares, and settle at t+1 mid prices.
    """
    try:
        import yfinance as yf
    except Exception:
        raise RuntimeError("simulate_backtest requires yfinance; install via pip install yfinance")

    # fetch price series for universe
    prices = {}
    for s in UNIVERSE:
        hist = yf.Ticker(s).history(period="1y")['Close'].dropna()
        prices[s] = hist

    # align dates
    common_idx = None
    for s, series in prices.items():
        if common_idx is None:
            common_idx = series.index
        else:
            common_idx = common_idx.intersection(series.index)
    common_idx = common_idx.sort_values()
    if len(common_idx) < 10:
        raise RuntimeError("Not enough overlapping price history to backtest; expand universe or timeframe.")

    equity = initial_capital
    equity_curve = []
    total_tc = 0.0
    total_trades = 0

    for i in range(len(common_idx) - 1):
        today = common_idx[i]
        tomorrow = common_idx[i + 1]
        now_ts = pd.Timestamp(today)

        # collect candidates across universe for today
        candidates = []
        for sym in UNIVERSE:
            S = float(prices[sym].loc[today])
            chain = fetch_option_chain(sym, now_ts=now_ts, S_override=S)
            if not chain:
                continue
            # filter
            filtered = []
            for q in chain:
                dte_days = (q.expiry - now_ts).days
                if dte_days < DTE_MIN or dte_days > DTE_MAX:
                    continue
                if q.open_interest < MIN_OI or q.volume < MIN_VOL:
                    continue
                if (q.bid is None or q.ask is None) or (q.ask - q.bid) / max(q.mid, 0.001) > 0.10:
                    continue
                filtered.append(q)
            if not filtered:
                continue

            # vectorized BS on filtered
            Ks = [q.strike for q in filtered]
            Ts = [days_to_expiry(q.expiry, now_ts) for q in filtered]
            opt_types = [q.option_type for q in filtered]
            bids = np.array([q.bid for q in filtered], dtype=float)
            asks = np.array([q.ask for q in filtered], dtype=float)
            mids = (bids + asks) / 2.0
            ivs = [q.iv if q.iv is not None else 0.25 for q in filtered]
            r_arr = np.full(len(Ks), RISK_FREE_RATE)
            S_arr = np.full(len(Ks), S)
            results = bs_price_and_greeks_vectorized(S_arr, Ks, r_arr, ivs, Ts, opt_types)
            bm_price = results['price']
            deltas = results['delta']
            spreads = asks - bids
            spreads = np.where(spreads <= 0, 1e-6, spreads)
            mispricing = mids - bm_price
            score = mispricing / spreads

            for i_q, q in enumerate(filtered):
                delta_abs = abs(deltas[i_q])
                if delta_abs < DELTA_BAND[0] or delta_abs > DELTA_BAND[1]:
                    continue
                candidates.append({
                    "underlying": sym,
                    "quote": q,
                    "score": float(score[i_q]),
                    "mid": float(mids[i_q]),
                    "bs_price": float(bm_price[i_q]),
                    "delta": float(deltas[i_q]),
                    "spread": float(spreads[i_q])
                })

        if not candidates:
            equity_curve.append((today, equity))
            continue

        cands_df = pd.DataFrame(candidates)
        cands_df['abs_score'] = cands_df['score'].abs()
        cands_df = cands_df.sort_values('abs_score', ascending=False).head(TOP_K)

        # execute and settle at tomorrow's prices
        for idx, row in cands_df.iterrows():
            q: OptionQuote = row['quote']
            side = 'buy' if row['bs_price'] - row['mid'] > max(0.01, row['spread'] * 0.5) else ('sell' if row['mid'] - row['bs_price'] > max(0.01, row['spread'] * 0.5) else None)
            if side is None:
                continue
            qty = 1
            notional = abs(row['mid']) * 100.0 * qty
            if notional > MAX_POSITION_NOTIONAL:
                continue

            # transaction costs and slippage
            tc = notional * tc_rate
            slippage = notional * slippage_pct
            total_tc += tc + slippage
            total_trades += 1

            # next day's option mid price: regenerate chain for tomorrow and lookup matching symbol (approx by strike+type+expiry)
            S_next = float(prices[row['underlying']].loc[tomorrow])
            chain_next = fetch_option_chain(row['underlying'], now_ts=pd.Timestamp(tomorrow), S_override=S_next)
            mid_next = None
            for q2 in chain_next:
                if (abs(q2.strike - q.strike) < 1e-6) and q2.option_type == q.option_type and q2.expiry == q.expiry:
                    mid_next = q2.mid
                    delta_next = q2.delta
                    break
            if mid_next is None:
                # fallback: assume option price moves proportionally to underlying via vega and delta approx
                mid_next = row['mid']  # assume no change
                delta_next = row['delta']

            sign = 1.0 if side == 'buy' else -1.0
            # option pnl (per contract = 100 * price change * sign)
            option_pnl = sign * (mid_next - row['mid']) * 100.0 * qty
            # hedge pnl: assume we hedge at day t using delta, and unwind day t+1 at delta_next (approx neutralizing exposure)
            hedge_shares = sign * row['delta'] * 100.0 * qty
            hedge_pnl = -hedge_shares * (S_next - S)  # hedge was opposite exposure
            # cost of hedging: assume tc on hedge notional
            hedge_tc = abs(hedge_shares) * S * tc_rate
            total_tc += hedge_tc

            pnl = option_pnl + hedge_pnl - tc - slippage - hedge_tc
            equity += pnl

        equity_curve.append((today, equity))

    equity_series = pd.Series(dict(equity_curve)).sort_index()
    return {"equity_curve": equity_series, "total_transaction_cost": total_tc, "total_trades": total_trades}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="Run a simple backtest simulation using historical underlying prices (requires yfinance).")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    args = parser.parse_args()

    if args.backtest:
        logger.info("Running backtest simulation...")
        res = simulate_backtest(initial_capital=args.initial_capital)
        eq = res['equity_curve']
        ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (252/len(eq)) - 1
        daily_rets = eq.pct_change().dropna()
        ann_vol = daily_rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float('nan')
        logger.info("Backtest: Return %.2f%%, Vol %.2f%%, Sharpe %.2f, Total trades %d, Total TC %.2f", ann_ret*100, ann_vol*100, sharpe, res['total_trades'], res['total_transaction_cost'])
        eq.to_csv("bs_backtest_equity.csv")
    else:
        logger.info("Starting Alpaca BS trader scaffold.")
        logger.info("IMPORTANT: You can run --backtest to simulate strategy behavior without a live options feed.")
        try:
            main_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down on user request.")
