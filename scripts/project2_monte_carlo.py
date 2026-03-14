import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.strategies import momentum_signal, sign_threshold_rule
from src.backtester import backtest_positions
from src.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

# Frozen champion parameters
LOOKBACK = 60
THRESHOLD = 0.001
TARGET_VOL = 0.10
VOL_WINDOW = 40
TCOST_BPS = 5.0

# Monte Carlo settings
N_SIMS = 100000
HORIZON_DAYS = 252   # about 1 trading year
RANDOM_SEED = 42


def apply_vol_targeting(log_returns: pd.Series, target_vol: float, window: int):
    rolling_vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    scale = (target_vol / rolling_vol).shift(1)
    scale = scale.replace([np.inf, -np.inf], np.nan).clip(upper=3.0)
    scale = scale.fillna(1.0)
    vt_returns = log_returns * scale
    return vt_returns


def build_champion_returns(prices: pd.DataFrame, log_returns: pd.DataFrame) -> pd.Series:
    signal = momentum_signal(prices, lookback=LOOKBACK)
    positions = sign_threshold_rule(signal, threshold=THRESHOLD)

    result = backtest_positions(
        asset_log_returns=log_returns,
        positions=positions,
        transaction_cost_bps=TCOST_BPS,
    )

    raw_returns = result.strategy_log_returns

    vt_returns = apply_vol_targeting(
        raw_returns,
        target_vol=TARGET_VOL,
        window=VOL_WINDOW,
    )

    return vt_returns.dropna()


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return drawdowns.min()


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

    champion_returns = build_champion_returns(prices, log_returns)

    # Bootstrap sample from historical daily champion returns
    hist_returns = champion_returns.values
    n_hist = len(hist_returns)

    sim_equities = np.zeros((HORIZON_DAYS + 1, N_SIMS))
    ending_equities = np.zeros(N_SIMS)
    sim_max_drawdowns = np.zeros(N_SIMS)

    for i in range(N_SIMS):
        sampled_returns = np.random.choice(hist_returns, size=HORIZON_DAYS, replace=True)

        equity_path = np.exp(np.cumsum(sampled_returns))
        equity_path = np.insert(equity_path, 0, 1.0)

        sim_equities[:, i] = equity_path
        ending_equities[i] = equity_path[-1]
        sim_max_drawdowns[i] = max_drawdown_from_equity(equity_path)

    # Summary stats
    prob_profit = np.mean(ending_equities > 1.0)
    prob_loss = np.mean(ending_equities < 1.0)
    prob_dd_10 = np.mean(sim_max_drawdowns <= -0.10)
    prob_dd_20 = np.mean(sim_max_drawdowns <= -0.20)

    p5 = np.percentile(sim_equities, 5, axis=1)
    p25 = np.percentile(sim_equities, 25, axis=1)
    p50 = np.percentile(sim_equities, 50, axis=1)
    p75 = np.percentile(sim_equities, 75, axis=1)
    p95 = np.percentile(sim_equities, 95, axis=1)

    print("\n=== Step 11: Monte Carlo Simulation of Champion Strategy ===\n")
    print(f"Number of simulations: {N_SIMS}")
    print(f"Horizon (trading days): {HORIZON_DAYS}")
    print(f"Probability of profit: {prob_profit:.2%}")
    print(f"Probability of loss:   {prob_loss:.2%}")
    print(f"Probability max drawdown <= -10%: {prob_dd_10:.2%}")
    print(f"Probability max drawdown <= -20%: {prob_dd_20:.2%}")
    print(f"Median ending equity: {np.median(ending_equities):.3f}")
    print(f"5th percentile ending equity: {np.percentile(ending_equities, 5):.3f}")
    print(f"95th percentile ending equity: {np.percentile(ending_equities, 95):.3f}")

    # Plot 1: sample paths + percentile bands
    plt.figure(figsize=(12, 6))

    # plot a subset of paths so the chart is readable
    for i in range(min(100, N_SIMS)):
        plt.plot(sim_equities[:, i], alpha=0.08)

    plt.plot(p50, linewidth=2, label="Median Path")
    plt.plot(p5, linestyle="--", linewidth=1, label="5th Percentile")
    plt.plot(p95, linestyle="--", linewidth=1, label="95th Percentile")

    plt.title("Step 11: Monte Carlo Simulated Equity Paths (1-Year Horizon)")
    plt.xlabel("Trading Days")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step11_monte_carlo_paths.png", dpi=150)
    plt.show()

    # Plot 2: ending equity histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ending_equities, bins=40, edgecolor="black")
    plt.axvline(1.0, linestyle="--", label="Break-even")
    plt.axvline(np.median(ending_equities), linestyle="-", label="Median")
    plt.title("Step 11: Distribution of Ending Equity")
    plt.xlabel("Ending Equity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step11_monte_carlo_hist.png", dpi=150)
    plt.show()

    print("\nSaved plots:")
    print("- step11_monte_carlo_paths.png")
    print("- step11_monte_carlo_hist.png")


if __name__ == "__main__":
    main()