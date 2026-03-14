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
from src.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

LOOKBACK = 60
THRESHOLD = 0.0
TCOST_BPS = 5.0

TARGET_VOL = 0.10
VOL_WINDOW = 20


def apply_vol_targeting(log_returns: pd.Series, target_vol=TARGET_VOL, window=VOL_WINDOW):
    """
    Scale daily log returns so realized vol tracks target_vol.
    Uses previous window's vol estimate to size today's exposure.
    """
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)

    vt_returns = log_returns * scale
    return vt_returns, rolling_vol, scale


def main():
    # 1. Load data
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

    # 2. Build positions from best signal found so far
    signal = momentum_signal(prices, lookback=LOOKBACK)
    positions = sign_threshold_rule(signal, threshold=THRESHOLD)

    # 3. Backtest original 60-day momentum strategy
    result_orig = backtest_positions(
        asset_log_returns=log_returns,
        positions=positions,
        transaction_cost_bps=TCOST_BPS,
    )

    orig_lr = result_orig.strategy_log_returns
    orig_eq = result_orig.equity_curve

    # 4. Apply vol targeting to original strategy returns
    vt_lr, rolling_vol, scale = apply_vol_targeting(
        orig_lr,
        target_vol=TARGET_VOL,
        window=VOL_WINDOW,
    )

    vt_eq = np.exp(vt_lr.cumsum())
    if len(vt_eq) > 0:
        vt_eq.iloc[0] = 1.0

    # 5. Summaries
    summary_orig = summarize_strategy("60d Momentum", orig_lr, orig_eq)
    summary_vt = summarize_strategy("60d Momentum + VT", vt_lr, vt_eq)

    summary_df = pd.DataFrame([summary_orig, summary_vt])

    print("\n=== Step 5: Vol Targeting on Best Signal (60-Day Momentum) ===")
    print(summary_df.to_string(index=False))

    # 6. Plot equity curves
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(orig_eq.index, orig_eq.values, label="60d Momentum")
    axes[0].plot(vt_eq.index, vt_eq.values, label="60d Momentum + VT")
    axes[0].set_title("Step 5: Equity Curves")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid(True)

    # Rolling vol
    axes[1].plot(rolling_vol.index, rolling_vol.values, color="orange")
    axes[1].axhline(TARGET_VOL, color="red", linestyle="--", label=f"Target {TARGET_VOL:.0%}")
    axes[1].set_title("Rolling Annualized Volatility")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()
    axes[1].grid(True)

    # Scaling factor
    axes[2].plot(scale.index, scale.values, color="green")
    axes[2].axhline(1.0, color="gray", linestyle="--", label="Scale = 1x")
    axes[2].set_title("Position Scaling Factor")
    axes[2].set_ylabel("Scale")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("step5_vol_targeting_lookback60.png", dpi=150)
    plt.show()

    print("\nPlot saved to step5_vol_targeting_lookback60.png")


if __name__ == "__main__":
    main()