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

TCOST_BPS = 5.0

# Original baseline
BASE_LOOKBACK = 20
BASE_THRESHOLD = 0.0

# Final champion
CHAMP_LOOKBACK = 60
CHAMP_THRESHOLD = 0.001
CHAMP_TARGET_VOL = 0.10
CHAMP_VOL_WINDOW = 40


def apply_vol_targeting(log_returns: pd.Series, target_vol: float, window: int):
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    vt_returns = log_returns * scale
    return vt_returns


def build_equal_weight_benchmark(asset_log_returns: pd.DataFrame):
    ew_returns = asset_log_returns.mean(axis=1)
    ew_equity = np.exp(ew_returns.cumsum())
    if len(ew_equity) > 0:
        ew_equity.iloc[0] = 1.0
    return ew_returns, ew_equity


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

    # 1. Equal-weight benchmark
    ew_returns, ew_equity = build_equal_weight_benchmark(log_returns)

    # 2. Original baseline strategy
    base_signal = momentum_signal(prices, lookback=BASE_LOOKBACK)
    base_positions = sign_threshold_rule(base_signal, threshold=BASE_THRESHOLD)

    base_result = backtest_positions(
        asset_log_returns=log_returns,
        positions=base_positions,
        transaction_cost_bps=TCOST_BPS,
    )

    base_returns = base_result.strategy_log_returns
    base_equity = base_result.equity_curve

    # 3. Final champion strategy
    champ_signal = momentum_signal(prices, lookback=CHAMP_LOOKBACK)
    champ_positions = sign_threshold_rule(champ_signal, threshold=CHAMP_THRESHOLD)

    champ_result = backtest_positions(
        asset_log_returns=log_returns,
        positions=champ_positions,
        transaction_cost_bps=TCOST_BPS,
    )

    champ_raw_returns = champ_result.strategy_log_returns
    champ_vt_returns = apply_vol_targeting(
        champ_raw_returns,
        target_vol=CHAMP_TARGET_VOL,
        window=CHAMP_VOL_WINDOW,
    )

    champ_equity = np.exp(champ_vt_returns.cumsum())
    if len(champ_equity) > 0:
        champ_equity.iloc[0] = 1.0

    # 4. Summaries
    summary_ew = summarize_strategy("Equal Weight Benchmark", ew_returns, ew_equity)
    summary_base = summarize_strategy("Original 20d Momentum", base_returns, base_equity)
    summary_champ = summarize_strategy("Champion Strategy", champ_vt_returns, champ_equity)

    summary_df = pd.DataFrame([summary_ew, summary_base, summary_champ])

    print("\n=== Step 8: Benchmark Comparison ===\n")
    print(summary_df.to_string(index=False))

    # 5. Plot equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(ew_equity.index, ew_equity.values, label="Equal Weight Benchmark")
    plt.plot(base_equity.index, base_equity.values, label="Original 20d Momentum")
    plt.plot(champ_equity.index, champ_equity.values, label="Champion Strategy")
    plt.title("Step 8: Champion vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step8_benchmark_comparison.png", dpi=150)
    plt.show()

    print("\nPlot saved to step8_benchmark_comparison.png")


if __name__ == "__main__":
    main()