import pandas as pd
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal, sign_threshold_rule
from src.backtester import backtest_positions
from src.metrics import summarize_strategy
from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW

LOOKBACKS = [5, 10, 20, 40, 60, 120]
THRESHOLD = 0.0
TCOST_BPS = 5.0


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

    for lookback in LOOKBACKS:
        signal = momentum_signal(prices, lookback=lookback)
        positions = sign_threshold_rule(signal, threshold=THRESHOLD)

        result = backtest_positions(
            asset_log_returns=log_returns,
            positions=positions,
            transaction_cost_bps=TCOST_BPS,
        )

        name = f"lookback={lookback}"
        summary = summarize_strategy(
            name,
            result.strategy_log_returns,
            result.equity_curve
        )

        summaries.append(summary)
        equity_curves[name] = result.equity_curve

    summary_df = pd.DataFrame(summaries)
    print("\n=== Step 4: Momentum Lookback Sweep ===")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    for name, eq in equity_curves.items():
        plt.plot(eq.index, eq.values, label=name)

    plt.title("Project 2 Step 4: Momentum Lookback Sweep")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step4_lookback_sweep.png", dpi=150)
    plt.show()

    print("\nPlot saved to step4_lookback_sweep.png")


if __name__ == "__main__":
    main()