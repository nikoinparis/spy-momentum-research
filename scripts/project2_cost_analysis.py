# scripts/project2_cost_analysis.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import momentum
from src.backtester import backtest_positions
from src.metrics import (
    annualized_return_from_log_returns,
    annualized_volatility_from_log_returns,
    sharpe_ratio_from_log_returns,
    max_drawdown,
)

MOM_LOOKBACK = 60
COST_BPS = 5.0  # start with 5 basis points per turnover event


def compute_turnover(positions: pd.DataFrame) -> pd.Series:
    # absolute change in position day to day
    pos_change = positions.diff().abs().fillna(0.0)

    # if multiple assets later, sum across columns
    turnover = pos_change.sum(axis=1)
    return turnover


def summarize(name: str, log_returns: pd.Series, equity_curve: pd.Series) -> dict:
    return {
        "Strategy": name,
        "Final Equity": float(equity_curve.iloc[-1]),
        "Annual Return": annualized_return_from_log_returns(log_returns),
        "Annual Vol": annualized_volatility_from_log_returns(log_returns),
        "Sharpe": sharpe_ratio_from_log_returns(log_returns),
        "Max Drawdown": max_drawdown(equity_curve),
    }


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(
        TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path
    )

    prices = data.prices.dropna()
    rets = data.log_returns.dropna()

    idx = prices.index.intersection(rets.index)
    prices = prices.loc[idx]
    rets = rets.loc[idx]

    positions = momentum(prices, lookback=MOM_LOOKBACK)
    turnover = compute_turnover(positions)

    gross_res = backtest_positions(rets, positions, transaction_cost_bps=0.0)
    net_res = backtest_positions(rets, positions, transaction_cost_bps=COST_BPS)

    gross_summary = summarize("Gross", gross_res.strategy_log_returns, gross_res.equity_curve)
    net_summary = summarize("Net", net_res.strategy_log_returns, net_res.equity_curve)

    summary_df = pd.DataFrame([gross_summary, net_summary]).set_index("Strategy")
    print("\nGross vs Net Performance\n")
    print(summary_df)

    print("\nTurnover Stats\n")
    print(f"Average daily turnover: {turnover.mean():.4f}")
    print(f"Annualized turnover (approx): {turnover.mean() * 252:.2f}")

    # Plot equity curves
    plt.figure(figsize=(12, 5))
    plt.plot(gross_res.equity_curve.index, gross_res.equity_curve.values, label="Gross")
    plt.plot(net_res.equity_curve.index, net_res.equity_curve.values, label=f"Net ({COST_BPS:.1f} bps)")
    plt.title("Project 2 Step 1: Gross vs Net Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot turnover
    plt.figure(figsize=(12, 4))
    plt.plot(turnover.index, turnover.values)
    plt.title("Project 2 Step 1: Daily Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()