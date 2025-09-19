import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# import your trader file (pretend it contains `api` and `UNIVERSE`)
from cvx_trader_v2 import UNIVERSE, START, END, fetch_alpaca_prices
import numpy as np

class RiskReturnAnalyzer:
    def __init__(self, price_data: pd.DataFrame):
        """
        price_data: DataFrame with datetime index and tickers as columns, values are prices.
        """
        self.price_data = price_data.sort_index()
        self.returns = self.price_data.pct_change().dropna()

    def compute_metrics(self):
        """Compute annualized mean return and volatility for each asset."""
        mean_returns = self.returns.mean() * 252 # trading days in a year
        volatilities = self.returns.std() * np.sqrt(252)
        return pd.DataFrame({
            'Return': mean_returns,
            'Volatility': volatilities
        })

    def plot_risk_return(self, ax=None):
        """Plot scatter of volatility vs return."""
        metrics = self.compute_metrics()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(metrics['Volatility'], metrics['Return'], s=50, alpha=0.7)

        for ticker in metrics.index:
            ax.annotate(ticker, (metrics.loc[ticker,'Volatility'], metrics.loc[ticker,'Return']),
                xytext=(5,5), textcoords="offset points", fontsize=8)

        ax.set_xlabel("Volatility (Annualized Std Dev)")
        ax.set_ylabel("Return (Annualized Mean)")
        ax.set_title("Risk-Return Plot")
        return ax

def main():
    # Fetch historical price data
    prices = fetch_alpaca_prices(UNIVERSE, start=START, end=END)

    # Compute risk/return metrics
    analyzer = RiskReturnAnalyzer(prices)
    metrics = analyzer.compute_metrics()
    print(metrics)

    # Plot
    ax = analyzer.plot_risk_return()
    plt.show()

if __name__ == "__main__":
    main()
