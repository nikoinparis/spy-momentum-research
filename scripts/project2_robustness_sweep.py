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

LOOKBACKS = [40, 60, 80]
THRESHOLDS = [0.0, 0.001, 0.002, 0.003, 0.005]
TARGET_VOLS = [0.08, 0.10, 0.12]
VOL_WINDOWS = [10, 20, 40]
TCOST_BPS = 5.0


def apply_vol_targeting(log_returns: pd.Series, target_vol: float, window: int):
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

    rows = []

    for lookback in LOOKBACKS:
        signal = momentum_signal(prices, lookback=lookback)

        for threshold in THRESHOLDS:
            positions = sign_threshold_rule(signal, threshold=threshold)

            result = backtest_positions(
                asset_log_returns=log_returns,
                positions=positions,
                transaction_cost_bps=TCOST_BPS,
            )

            base_returns = result.strategy_log_returns

            for target_vol in TARGET_VOLS:
                for vol_window in VOL_WINDOWS:
                    vt_returns = apply_vol_targeting(
                        base_returns,
                        target_vol=target_vol,
                        window=vol_window,
                    )

                    vt_equity = np.exp(vt_returns.cumsum())
                    if len(vt_equity) > 0:
                        vt_equity.iloc[0] = 1.0

                    summary = summarize_strategy(
                        f"lb={lookback},thr={threshold},tv={target_vol},vw={vol_window}",
                        vt_returns,
                        vt_equity,
                    )

                    row = {
                        "lookback": lookback,
                        "threshold": threshold,
                        "target_vol": target_vol,
                        "vol_window": vol_window,
                        "final_equity": summary["Final Equity"],
                        "annual_return": summary["Annual Return"],
                        "annual_vol": summary["Annual Vol"],
                        "sharpe": summary["Sharpe"],
                        "max_drawdown": summary["Max Drawdown"],
                        "win_rate": summary["Win Rate"],
                    }
                    rows.append(row)

    results_df = pd.DataFrame(rows)

    # Sort by Sharpe first, then Final Equity
    ranked_df = results_df.sort_values(
        by=["sharpe", "final_equity"],
        ascending=[False, False]
    ).reset_index(drop=True)

    print("\n=== Step 7: Robustness Sweep Around Winning Strategy ===")
    print("\nTop 15 by Sharpe:\n")
    print(ranked_df.head(15).to_string(index=False))

    results_df.to_csv("step7_robustness_results.csv", index=False)
    print("\nSaved full results to step7_robustness_results.csv")

    # Simple visualization: average Sharpe by lookback
    sharpe_by_lookback = results_df.groupby("lookback")["sharpe"].mean()

    plt.figure(figsize=(8, 5))
    plt.plot(sharpe_by_lookback.index, sharpe_by_lookback.values, marker="o")
    plt.title("Average Sharpe by Lookback")
    plt.xlabel("Lookback")
    plt.ylabel("Average Sharpe")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step7_avg_sharpe_by_lookback.png", dpi=150)
    plt.show()

    print("Saved plot to step7_avg_sharpe_by_lookback.png")


if __name__ == "__main__":
    main()