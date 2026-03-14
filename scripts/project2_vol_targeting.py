import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Make sure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal, sign_threshold_rule
from src.backtester import backtest_positions
from src.metrics import summarize_strategy
from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW

TARGET_VOL = 0.10
VOL_WINDOW = 20


def apply_vol_targeting(log_returns: pd.Series, target_vol=TARGET_VOL, window=VOL_WINDOW):
    """
    Scale daily log returns so realized vol tracks target_vol.
    Uses previous window's vol to size today's position (no lookahead).
    """
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    vt_returns = log_returns * scale
    return vt_returns, rolling_vol, scale


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    cache_path = os.path.join(DATA_DIR_RAW, "prices.csv")
    price_data = get_price_data(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        price_field=PRICE_FIELD,
        interval=INTERVAL,
        cache_path=cache_path,
    )
    prices     = price_data.prices
    log_returns = price_data.log_returns

    # ── 2. Build positions from momentum signal ───────────────────────────────
    signal    = momentum_signal(prices, lookback=20)
    positions = sign_threshold_rule(signal, threshold=0.0)

    # ── 3. Backtest original (no vol targeting) ───────────────────────────────
    result_orig = backtest_positions(
        asset_log_returns=log_returns,
        positions=positions,
        transaction_cost_bps=5.0,
    )
    orig_lr  = result_orig.strategy_log_returns
    orig_eq  = result_orig.equity_curve

    # ── 4. Vol-targeted backtest ──────────────────────────────────────────────
    # Apply vol targeting to the raw strategy log returns
    vt_lr, rolling_vol, scale = apply_vol_targeting(orig_lr, TARGET_VOL, VOL_WINDOW)

    # Rebuild equity curve from vol-targeted log returns
    vt_eq = np.exp(vt_lr.cumsum())
    vt_eq.iloc[0] = 1.0

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    print("\n── Original Strategy Metrics ──")
    print(summarize_strategy("Original", orig_lr, orig_eq).to_string())

    print("\n── Vol-Targeted Strategy Metrics ──")
    print(summarize_strategy("Vol Targeted", vt_lr, vt_eq).to_string())

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Equity curves
    axes[0].plot(orig_eq.index, orig_eq.values, label="Original")
    axes[0].plot(vt_eq.index,   vt_eq.values,   label="Vol Targeted")
    axes[0].set_title("Project 2 Step 2: Volatility Targeting — Equity Curves")
    axes[0].set_ylabel("Equity (starts at 1.0)")
    axes[0].legend()
    axes[0].grid(True)

    # Rolling vol
    axes[1].plot(rolling_vol.index, rolling_vol.values, color="orange")
    axes[1].axhline(TARGET_VOL, color="red", linestyle="--", label=f"Target {TARGET_VOL:.0%}")
    axes[1].set_title("Estimated Rolling Annualized Volatility")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()
    axes[1].grid(True)

    # Scale factor
    axes[2].plot(scale.index, scale.values, color="green")
    axes[2].axhline(1.0, color="gray", linestyle="--", label="Scale = 1x")
    axes[2].set_title("Position Scaling Factor (capped at 3x)")
    axes[2].set_ylabel("Scale")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("vol_targeting_output.png", dpi=150)
    plt.show()
    print("\nPlot saved to vol_targeting_output.png")


if __name__ == "__main__":
    main()