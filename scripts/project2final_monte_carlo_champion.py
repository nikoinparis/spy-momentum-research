import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal
from src.backtester import backtest_positions
from src.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

# Step 15 champion settings
LOOKBACK = 60
THRESHOLD = 0.001
MA_WINDOW = 200
TARGET_VOL = 0.10
VOL_WINDOW = 40
TCOST_BPS = 5.0

# Monte Carlo settings
N_SIMS = 100000
HORIZONS = [252]
RANDOM_SEED = 42


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


def build_step15_returns(prices: pd.DataFrame, log_returns: pd.DataFrame) -> pd.Series:
    signal = momentum_signal(prices, lookback=LOOKBACK)
    positions = long_only_rule(signal, THRESHOLD)
    positions = apply_defensive_filter(positions, prices, MA_WINDOW)

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

    return vt_returns.dropna()


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return drawdowns.min()


def run_bootstrap_monte_carlo(hist_returns: np.ndarray, horizon_days: int, n_sims: int):
    sim_equities = np.zeros((horizon_days + 1, n_sims))
    ending_equities = np.zeros(n_sims)
    sim_max_drawdowns = np.zeros(n_sims)

    for i in range(n_sims):
        sampled_returns = np.random.choice(hist_returns, size=horizon_days, replace=True)
        equity_path = np.exp(np.cumsum(sampled_returns))
        equity_path = np.insert(equity_path, 0, 1.0)

        sim_equities[:, i] = equity_path
        ending_equities[i] = equity_path[-1]
        sim_max_drawdowns[i] = max_drawdown_from_equity(equity_path)

    return sim_equities, ending_equities, sim_max_drawdowns


def summarize_mc(ending_equities: np.ndarray, sim_max_drawdowns: np.ndarray):
    return {
        "prob_profit": np.mean(ending_equities > 1.0),
        "prob_loss": np.mean(ending_equities < 1.0),
        "prob_dd_10": np.mean(sim_max_drawdowns <= -0.10),
        "prob_dd_20": np.mean(sim_max_drawdowns <= -0.20),
        "median_end": np.median(ending_equities),
        "p5_end": np.percentile(ending_equities, 5),
        "p95_end": np.percentile(ending_equities, 95),
    }


def plot_mc_results(sim_equities, ending_equities, horizon_days, suffix):
    p5 = np.percentile(sim_equities, 5, axis=1)
    p50 = np.percentile(sim_equities, 50, axis=1)
    p95 = np.percentile(sim_equities, 95, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(min(100, sim_equities.shape[1])):
        plt.plot(sim_equities[:, i], alpha=0.06)

    plt.plot(p50, linewidth=2, label="Median Path")
    plt.plot(p5, linestyle="--", linewidth=1, label="5th Percentile")
    plt.plot(p95, linestyle="--", linewidth=1, label="95th Percentile")

    plt.title(f"Step 17: Monte Carlo Paths ({horizon_days} Trading Days)")
    plt.xlabel("Trading Days")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"step17_mc_paths_{suffix}.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(ending_equities, bins=40, edgecolor="black")
    plt.axvline(1.0, linestyle="--", label="Break-even")
    plt.axvline(np.median(ending_equities), linestyle="-", label="Median")
    plt.title(f"Step 17: Ending Equity Distribution ({horizon_days} Trading Days)")
    plt.xlabel("Ending Equity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"step17_mc_hist_{suffix}.png", dpi=150)
    plt.show()


def main():
    np.random.seed(RANDOM_SEED)

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

    champion_returns = build_step15_returns(prices, log_returns)
    hist_returns = champion_returns.values

    print("\n=== Step 17: Monte Carlo on Step 15 Champion ===\n")
    print("Strategy: Long-only momentum + 200d MA filter + vol targeting")
    print(f"Simulations: {N_SIMS}\n")

    for horizon in HORIZONS:
        sim_equities, ending_equities, sim_max_drawdowns = run_bootstrap_monte_carlo(
            hist_returns=hist_returns,
            horizon_days=horizon,
            n_sims=N_SIMS,
        )

        stats = summarize_mc(ending_equities, sim_max_drawdowns)

        print(f"--- Horizon: {horizon} trading days ---")
        print(f"Probability of profit: {stats['prob_profit']:.2%}")
        print(f"Probability of loss:   {stats['prob_loss']:.2%}")
        print(f"Probability max drawdown <= -10%: {stats['prob_dd_10']:.2%}")
        print(f"Probability max drawdown <= -20%: {stats['prob_dd_20']:.2%}")
        print(f"Median ending equity: {stats['median_end']:.3f}")
        print(f"5th percentile ending equity: {stats['p5_end']:.3f}")
        print(f"95th percentile ending equity: {stats['p95_end']:.3f}\n")

        plot_mc_results(
            sim_equities=sim_equities,
            ending_equities=ending_equities,
            horizon_days=horizon,
            suffix=str(horizon),
        )

    print("Saved plot files:")
    print("- step17_mc_paths_50.png")
    print("- step17_mc_hist_50.png")
    print("- step17_mc_paths_252.png")
    print("- step17_mc_hist_252.png")


if __name__ == "__main__":
    main()