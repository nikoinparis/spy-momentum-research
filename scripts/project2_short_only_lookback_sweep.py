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

SHORT_LOOKBACKS = [5, 10, 20, 40, 60, 90, 100]
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


def short_only_rule(signal: pd.DataFrame, threshold: float) -> pd.DataFrame:
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    positions[signal < -threshold] = -1.0
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

    summaries = []
    equity_curves = {}

    for lb in SHORT_LOOKBACKS:
        signal = momentum_signal(prices, lookback=lb)
        positions = short_only_rule(signal, THRESHOLD)

        result = backtest_positions(
            asset_log_returns=log_returns,
            positions=positions,
            transaction_cost_bps=TCOST_BPS,
        )

        vt_returns = apply_vol_targeting(
            result.strategy_log_returns,
            target_vol=TARGET_VOL,
            window=VOL_WINDOW,
        )

        equity = np.exp(vt_returns.cumsum())
        if len(equity) > 0:
            equity.iloc[0] = 1.0

        name = f"short_only_lb={lb}"
        summary = summarize_strategy(name, vt_returns, equity)

        summaries.append(summary)
        equity_curves[name] = equity

    summary_df = pd.DataFrame(summaries)
    print("\n=== Step 13: Short-Only Lookback Sweep ===\n")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    for name, eq in equity_curves.items():
        plt.plot(eq.index, eq.values, label=name)

    plt.title("Step 13: Short-Only Lookback Sweep")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step13_short_only_lookback_sweep.png", dpi=150)
    plt.show()

    print("\nPlot saved to step13_short_only_lookback_sweep.png")


if __name__ == "__main__":
    main()