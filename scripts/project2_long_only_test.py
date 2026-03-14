import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal
from src.backtester import backtest_positions
from src.metrics import summarize_strategy
from src.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

# Current champion settings
LOOKBACK = 60
THRESHOLD = 0.001
TARGET_VOL = 0.10
VOL_WINDOW = 40
TCOST_BPS = 5.0


def apply_vol_targeting(log_returns: pd.Series, target_vol: float, window: int):
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    return log_returns * scale


def long_short_rule(signal: pd.DataFrame, threshold: float) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    positions[signal > threshold] = 1.0
    positions[signal < -threshold] = -1.0
    return positions


def long_only_rule(signal: pd.DataFrame, threshold: float) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    positions[signal > threshold] = 1.0
    return positions


def main():
    cache_path = os.path.join(DATA_DIR_RAW, "prices.csv")
    price_data = get_price_data(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        price_field=PRICE_FIELD,
        interval=INTERVAL,
        cache_path=cache_path,
    )

    prices = price_data.prices
    log_returns = price_data.log_returns

    signal = momentum_signal(prices, lookback=LOOKBACK)

    # Long-short champion
    ls_positions = long_short_rule(signal, THRESHOLD)
    ls_result = backtest_positions(
        asset_log_returns=log_returns,
        positions=ls_positions,
        transaction_cost_bps=TCOST_BPS,
    )
    ls_returns = apply_vol_targeting(ls_result.strategy_log_returns, TARGET_VOL, VOL_WINDOW)
    ls_equity = np.exp(ls_returns.cumsum())
    ls_equity.iloc[0] = 1.0

    # Long-only version
    lo_positions = long_only_rule(signal, THRESHOLD)
    lo_result = backtest_positions(
        asset_log_returns=log_returns,
        positions=lo_positions,
        transaction_cost_bps=TCOST_BPS,
    )
    lo_returns = apply_vol_targeting(lo_result.strategy_log_returns, TARGET_VOL, VOL_WINDOW)
    lo_equity = np.exp(lo_returns.cumsum())
    lo_equity.iloc[0] = 1.0

    summary_ls = summarize_strategy("Long-Short Champion", ls_returns, ls_equity)
    summary_lo = summarize_strategy("Long-Only Version", lo_returns, lo_equity)

    summary_df = pd.DataFrame([summary_ls, summary_lo])

    print("\n=== Step 12: Long-Only vs Long-Short ===\n")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    plt.plot(ls_equity.index, ls_equity.values, label="Long-Short Champion")
    plt.plot(lo_equity.index, lo_equity.values, label="Long-Only Version")
    plt.title("Step 12: Long-Only vs Long-Short")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step12_long_only_test.png", dpi=150)
    plt.show()

    print("\nPlot saved to step12_long_only_test.png")


if __name__ == "__main__":
    main()