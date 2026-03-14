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

LOOKBACK = 60
THRESHOLDS = [0.0, 0.002, 0.005, 0.01]
TCOST_BPS = 5.0

TARGET_VOL = 0.10
VOL_WINDOW = 20


def apply_vol_targeting(log_returns: pd.Series, target_vol=TARGET_VOL, window=VOL_WINDOW):
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    vt_returns = log_returns * scale
    return vt_returns


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

    summaries = []
    equity_curves = {}

    for threshold in THRESHOLDS:
        positions = sign_threshold_rule(signal, threshold=threshold)

        result = backtest_positions(
            asset_log_returns=log_returns,
            positions=positions,
            transaction_cost_bps=TCOST_BPS,
        )

        vt_returns = apply_vol_targeting(result.strategy_log_returns)
        vt_equity = np.exp(vt_returns.cumsum())
        if len(vt_equity) > 0:
            vt_equity.iloc[0] = 1.0

        name = f"60d+VT thr={threshold}"
        summary = summarize_strategy(name, vt_returns, vt_equity)

        summaries.append(summary)
        equity_curves[name] = vt_equity

    summary_df = pd.DataFrame(summaries)
    print("\n=== Step 6: Threshold Filter on Best Strategy ===")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    for name, eq in equity_curves.items():
        plt.plot(eq.index, eq.values, label=name)

    plt.title("Step 6: Threshold Filter on 60-Day Momentum + Vol Targeting")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step6_threshold_on_best_strategy.png", dpi=150)
    plt.show()

    print("\nPlot saved to step6_threshold_on_best_strategy.png")


if __name__ == "__main__":
    main()