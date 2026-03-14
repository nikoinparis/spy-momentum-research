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
    END_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

# Champion parameters from Step 8
LOOKBACK = 60
THRESHOLD = 0.001
TARGET_VOL = 0.10
VOL_WINDOW = 40
TCOST_BPS = 5.0

# Train/test split
SPLIT_DATE = "2021-01-01"


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

    # Full-sample champion
    full_returns, full_equity = build_champion_returns(prices, log_returns)

    # Train/test split masks
    train_mask = full_returns.index < SPLIT_DATE
    test_mask = full_returns.index >= SPLIT_DATE

    train_returns = full_returns.loc[train_mask]
    test_returns = full_returns.loc[test_mask]

    train_equity = np.exp(train_returns.cumsum())
    test_equity = np.exp(test_returns.cumsum())

    if len(train_equity) > 0:
        train_equity.iloc[0] = 1.0
    if len(test_equity) > 0:
        test_equity.iloc[0] = 1.0

    # Summaries
    summary_train = summarize_strategy("Train Period", train_returns, train_equity)
    summary_test = summarize_strategy("Test Period", test_returns, test_equity)
    summary_full = summarize_strategy("Full Sample", full_returns, full_equity)

    summary_df = pd.DataFrame([summary_train, summary_test, summary_full])

    print("\n=== Step 9: Train/Test Validation of Champion Strategy ===\n")
    print(f"Split date: {SPLIT_DATE}\n")
    print(summary_df.to_string(index=False))

    # Plot full equity with train/test divider
    plt.figure(figsize=(12, 6))
    plt.plot(full_equity.index, full_equity.values, label="Champion Strategy")
    plt.axvline(pd.Timestamp(SPLIT_DATE), color="red", linestyle="--", label="Train/Test Split")
    plt.title("Step 9: Train/Test Validation")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step9_train_test_validation.png", dpi=150)
    plt.show()

    print("\nPlot saved to step9_train_test_validation.png")


if __name__ == "__main__":
    main()