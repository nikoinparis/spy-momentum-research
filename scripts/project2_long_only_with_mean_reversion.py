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

# Core strategy settings from Step 15
LOOKBACK = 60
THRESHOLD = 0.001
TARGET_VOL = 0.10
VOL_WINDOW = 40
TCOST_BPS = 5.0
MA_WINDOW = 200

# Mean reversion overlay settings
MR_LOOKBACK = 5
MR_TRIGGER = -0.04   # oversold if 5-day return <= -4%


def apply_vol_targeting(log_returns: pd.Series, target_vol: float, window: int):
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    return log_returns * scale


def long_only_rule(signal: pd.DataFrame, threshold: float) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    positions[signal > threshold] = 1.0
    return positions


def apply_defensive_filter(positions: pd.DataFrame, prices: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    moving_avg = prices.rolling(ma_window).mean()
    bullish_regime = prices > moving_avg
    return positions.where(bullish_regime, 0.0)


def oversold_rebound_overlay(prices: pd.DataFrame, ma_window: int, mr_lookback: int, mr_trigger: float) -> pd.DataFrame:
    """
    Mean-reversion overlay:
    - only active in bullish regime (price > 200d MA)
    - if short-term return is very negative, turn long for rebound
    """
    moving_avg = prices.rolling(ma_window).mean()
    bullish_regime = prices > moving_avg

    short_term_return = prices.pct_change(mr_lookback)
    oversold = short_term_return <= mr_trigger

    overlay = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    overlay[bullish_regime & oversold] = 1.0
    return overlay


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

    # ----- Step 15 winner: long-only momentum + defensive filter -----
    signal = momentum_signal(prices, lookback=LOOKBACK)
    base_positions = long_only_rule(signal, THRESHOLD)
    filt_positions = apply_defensive_filter(base_positions, prices, MA_WINDOW)

    filt_result = backtest_positions(
        asset_log_returns=log_returns,
        positions=filt_positions,
        transaction_cost_bps=TCOST_BPS,
    )

    filt_returns = apply_vol_targeting(
        filt_result.strategy_log_returns,
        target_vol=TARGET_VOL,
        window=VOL_WINDOW,
    )

    filt_equity = np.exp(filt_returns.cumsum())
    if len(filt_equity) > 0:
        filt_equity.iloc[0] = 1.0

    # ----- Step 16 candidate: add oversold rebound overlay -----
    overlay_positions = oversold_rebound_overlay(
        prices=prices,
        ma_window=MA_WINDOW,
        mr_lookback=MR_LOOKBACK,
        mr_trigger=MR_TRIGGER,
    )

    # Union of the core momentum long signal and the oversold rebound long signal
    combo_positions = pd.DataFrame(
        np.maximum(filt_positions.values, overlay_positions.values),
        index=prices.index,
        columns=prices.columns,
    )

    combo_result = backtest_positions(
        asset_log_returns=log_returns,
        positions=combo_positions,
        transaction_cost_bps=TCOST_BPS,
    )

    combo_returns = apply_vol_targeting(
        combo_result.strategy_log_returns,
        target_vol=TARGET_VOL,
        window=VOL_WINDOW,
    )

    combo_equity = np.exp(combo_returns.cumsum())
    if len(combo_equity) > 0:
        combo_equity.iloc[0] = 1.0

    # ----- Summaries -----
    summary_base = summarize_strategy("Long-Only + 200d MA Filter", filt_returns, filt_equity)
    summary_combo = summarize_strategy("Step 16: + Oversold Rebound", combo_returns, combo_equity)

    summary_df = pd.DataFrame([summary_base, summary_combo])

    print("\n=== Step 16: Long-Only Momentum + Mean Reversion Overlay ===\n")
    print(summary_df.to_string(index=False))

    # ----- Plot -----
    plt.figure(figsize=(12, 6))
    plt.plot(filt_equity.index, filt_equity.values, label="Long-Only + 200d MA Filter")
    plt.plot(combo_equity.index, combo_equity.values, label="Step 16: + Oversold Rebound")
    plt.title("Step 16: Defensive Long-Only vs Mean-Reversion Overlay")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step16_long_only_mean_reversion.png", dpi=150)
    plt.show()

    print("\nPlot saved to step16_long_only_mean_reversion.png")


if __name__ == "__main__":
    main()