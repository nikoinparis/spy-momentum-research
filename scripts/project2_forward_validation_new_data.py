import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal, sign_threshold_rule
from src.backtester import backtest_positions
from src.metrics import summarize_strategy
from src.config import (
    TICKERS,
    START_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

# Frozen champion parameters from earlier steps
LOOKBACK = 60
THRESHOLD = 0.001
TARGET_VOL = 0.10
VOL_WINDOW = 40
TCOST_BPS = 5.0

# Old sample end date (the point after which data is considered "new")
OLD_SAMPLE_END = "2025-12-31"


def apply_vol_targeting(log_returns: pd.Series, target_vol: float, window: int):
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    vt_returns = log_returns * scale
    return vt_returns


def build_champion_returns(prices: pd.DataFrame, log_returns: pd.DataFrame):
    signal = momentum_signal(prices, lookback=LOOKBACK)
    positions = sign_threshold_rule(signal, threshold=THRESHOLD)

    result = backtest_positions(
        asset_log_returns=log_returns,
        positions=positions,
        transaction_cost_bps=TCOST_BPS,
    )

    raw_returns = result.strategy_log_returns

    vt_returns = apply_vol_targeting(
        raw_returns,
        target_vol=TARGET_VOL,
        window=VOL_WINDOW,
    )

    equity = np.exp(vt_returns.cumsum())
    if len(equity) > 0:
        equity.iloc[0] = 1.0

    return vt_returns, equity


def main():
    # Save refreshed extended data to a new file so old cache stays untouched
    cache_path = os.path.join(DATA_DIR_RAW, "prices_extended.csv")

    # end=None means "download through latest available date"
    price_data = get_price_data(
        tickers=TICKERS,
        start=START_DATE,
        end=None,
        price_field=PRICE_FIELD,
        interval=INTERVAL,
        cache_path=cache_path,
        force_download=True,   # important: force fresh download
    )

    prices = price_data.prices
    log_returns = price_data.log_returns

    # Build frozen champion on extended dataset
    full_returns, full_equity = build_champion_returns(prices, log_returns)

    # Split into old sample vs newly downloaded extension
    old_mask = full_returns.index <= OLD_SAMPLE_END
    new_mask = full_returns.index > OLD_SAMPLE_END

    old_returns = full_returns.loc[old_mask]
    new_returns = full_returns.loc[new_mask]

    old_equity = np.exp(old_returns.cumsum())
    new_equity = np.exp(new_returns.cumsum())

    if len(old_equity) > 0:
        old_equity.iloc[0] = 1.0
    if len(new_equity) > 0:
        new_equity.iloc[0] = 1.0

    summary_old = summarize_strategy("Old Sample", old_returns, old_equity)

    summaries = [summary_old]

    if len(new_returns) > 0:
        summary_new = summarize_strategy("New Unseen Data", new_returns, new_equity)
        summaries.append(summary_new)
    else:
        print("\nNo data exists after OLD_SAMPLE_END. Try changing the cutoff date.")
        summary_new = None

    summary_full = summarize_strategy("Full Extended Sample", full_returns, full_equity)
    summaries.append(summary_full)

    summary_df = pd.DataFrame(summaries)

    print("\n=== Step 10: Forward Validation on Newly Downloaded Data ===\n")
    print(f"Old sample end: {OLD_SAMPLE_END}")
    print(f"Latest downloaded date: {full_returns.index.max().date()}\n")
    print(summary_df.to_string(index=False))

    # Plot full equity with boundary line
    plt.figure(figsize=(12, 6))
    plt.plot(full_equity.index, full_equity.values, label="Champion Strategy")
    plt.axvline(pd.Timestamp(OLD_SAMPLE_END), color="red", linestyle="--", label="Old/New Boundary")
    plt.title("Step 10: Forward Validation on Newly Downloaded Data")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step10_forward_validation_new_data.png", dpi=150)
    plt.show()

    print("\nPlot saved to step10_forward_validation_new_data.png")


if __name__ == "__main__":
    main()