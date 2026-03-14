import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal, sign_threshold_rule
from src.backtester import backtest_positions
from src.metrics import summarize_strategy
from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW

LOOKBACK = 20
THRESHOLDS = [0.0, 0.002, 0.005, 0.01]


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
            transaction_cost_bps=5.0,
        )

        strat_name = f"threshold={threshold}"
        summary = summarize_strategy(
            strat_name,
            result.strategy_log_returns,
            result.equity_curve
        )

        summaries.append(summary)
        equity_curves[strat_name] = result.equity_curve

    summary_df = pd.DataFrame(summaries)
    print("\n=== Step 3: Momentum Threshold Filter Comparison ===")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    for name, eq in equity_curves.items():
        plt.plot(eq.index, eq.values, label=name)

    plt.title("Project 2 Step 3: Momentum Threshold Filter")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step3_threshold_filter.png", dpi=150)
    plt.show()

    print("\nPlot saved to step3_threshold_filter.png")


if __name__ == "__main__":
    main()