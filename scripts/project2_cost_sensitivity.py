# scripts/project2_cost_sensitivity.py
from __future__ import annotations

import pandas as pd

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
COST_LEVELS_BPS = [0.0, 2.0, 5.0, 10.0, 20.0]


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

    rows = []
    for cost_bps in COST_LEVELS_BPS:
        res = backtest_positions(rets, positions, transaction_cost_bps=cost_bps)
        rows.append({
            "Cost (bps)": cost_bps,
            "Final Equity": float(res.equity_curve.iloc[-1]),
            "Annual Return": annualized_return_from_log_returns(res.strategy_log_returns),
            "Annual Vol": annualized_volatility_from_log_returns(res.strategy_log_returns),
            "Sharpe": sharpe_ratio_from_log_returns(res.strategy_log_returns),
            "Max Drawdown": max_drawdown(res.equity_curve),
        })

    df = pd.DataFrame(rows).set_index("Cost (bps)")
    print("\nCost Sensitivity Analysis\n")
    print(df)


if __name__ == "__main__":
    main()