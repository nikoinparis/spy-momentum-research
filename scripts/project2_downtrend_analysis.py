import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_price_data
from src.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    INTERVAL,
    DATA_DIR_RAW,
)

# We assume SPY is in your ticker list and use the first column if only one exists.
TICKER_TO_ANALYZE = "SPY"
DRAWDOWN_THRESHOLD = 0.05  # 5% drawdown from prior peak


def get_series_from_prices(prices: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker in prices.columns:
        s = prices[ticker].copy()
    else:
        # fallback: if only one column exists, use it
        if prices.shape[1] == 1:
            s = prices.iloc[:, 0].copy()
        else:
            raise ValueError(f"{ticker} not found in price columns: {list(prices.columns)}")
    return s.dropna()


def find_downtrend_episodes(price_series: pd.Series, threshold: float = DRAWDOWN_THRESHOLD):
    """
    Define a downtrend episode as:
    - drawdown from running peak >= threshold
    - episode starts when threshold is first crossed
    - episode ends when price recovers to a new peak (drawdown back to 0)
    """
    running_peak = price_series.cummax()
    drawdown = price_series / running_peak - 1.0

    episodes = []
    in_episode = False
    start_date = None
    trough_date = None
    peak_before_episode = None
    max_drawdown = None

    prev_date = None

    for date, dd in drawdown.items():
        if (not in_episode) and (dd <= -threshold):
            in_episode = True
            start_date = date
            trough_date = date
            peak_before_episode = running_peak.loc[date]
            max_drawdown = dd

        elif in_episode:
            if dd < max_drawdown:
                max_drawdown = dd
                trough_date = date

            # episode ends when price fully recovers to its running peak again
            if np.isclose(dd, 0.0) or dd == 0.0:
                end_date = date

                start_price = price_series.loc[start_date]
                trough_price = price_series.loc[trough_date]
                end_price = price_series.loc[end_date]

                duration_days = (end_date - start_date).days
                trading_days = price_series.loc[start_date:end_date].shape[0] - 1
                trough_trading_days = price_series.loc[start_date:trough_date].shape[0] - 1

                episode_return = end_price / start_price - 1.0
                decline_from_start_to_trough = trough_price / start_price - 1.0

                episodes.append({
                    "start_date": start_date,
                    "trough_date": trough_date,
                    "end_date": end_date,
                    "duration_days": duration_days,
                    "trading_days": trading_days,
                    "days_to_trough": trough_trading_days,
                    "start_price": start_price,
                    "trough_price": trough_price,
                    "end_price": end_price,
                    "max_drawdown": max_drawdown,
                    "return_start_to_trough": decline_from_start_to_trough,
                    "return_start_to_end": episode_return,
                })

                in_episode = False
                start_date = None
                trough_date = None
                peak_before_episode = None
                max_drawdown = None

        prev_date = date

    # Handle unfinished episode at end of sample
    if in_episode:
        end_date = price_series.index[-1]

        start_price = price_series.loc[start_date]
        trough_price = price_series.loc[trough_date]
        end_price = price_series.loc[end_date]

        duration_days = (end_date - start_date).days
        trading_days = price_series.loc[start_date:end_date].shape[0] - 1
        trough_trading_days = price_series.loc[start_date:trough_date].shape[0] - 1

        episode_return = end_price / start_price - 1.0
        decline_from_start_to_trough = trough_price / start_price - 1.0

        episodes.append({
            "start_date": start_date,
            "trough_date": trough_date,
            "end_date": end_date,
            "duration_days": duration_days,
            "trading_days": trading_days,
            "days_to_trough": trough_trading_days,
            "start_price": start_price,
            "trough_price": trough_price,
            "end_price": end_price,
            "max_drawdown": max_drawdown,
            "return_start_to_trough": decline_from_start_to_trough,
            "return_start_to_end": episode_return,
            "unfinished": True,
        })

    return pd.DataFrame(episodes), drawdown


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()

    summary = {
        "num_episodes": len(episodes),
        "avg_trading_days": episodes["trading_days"].mean(),
        "median_trading_days": episodes["trading_days"].median(),
        "avg_days_to_trough": episodes["days_to_trough"].mean(),
        "median_days_to_trough": episodes["days_to_trough"].median(),
        "avg_max_drawdown": episodes["max_drawdown"].mean(),
        "median_max_drawdown": episodes["max_drawdown"].median(),
        "avg_start_to_trough_return": episodes["return_start_to_trough"].mean(),
        "median_start_to_trough_return": episodes["return_start_to_trough"].median(),
    }

    return pd.DataFrame([summary])


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
    price_series = get_series_from_prices(prices, TICKER_TO_ANALYZE)

    episodes_df, drawdown = find_downtrend_episodes(
        price_series,
        threshold=DRAWDOWN_THRESHOLD,
    )

    summary_df = summarize_episodes(episodes_df)

    print("\n=== Step 14: Downtrend Episode Analysis ===\n")
    print(f"Ticker analyzed: {TICKER_TO_ANALYZE}")
    print(f"Drawdown threshold: {DRAWDOWN_THRESHOLD:.0%}\n")

    if episodes_df.empty:
        print("No downtrend episodes found.")
        return

    print("Summary:")
    print(summary_df.to_string(index=False))

    print("\nEpisodes:")
    print(episodes_df.to_string(index=False))

    episodes_df.to_csv("step14_downtrend_episodes.csv", index=False)
    print("\nSaved episode table to step14_downtrend_episodes.csv")

    # Plot price with shaded downtrend episodes
    plt.figure(figsize=(12, 6))
    plt.plot(price_series.index, price_series.values, label=TICKER_TO_ANALYZE)

    for _, row in episodes_df.iterrows():
        plt.axvspan(
            pd.to_datetime(row["start_date"]),
            pd.to_datetime(row["end_date"]),
            alpha=0.2
        )

    plt.title(f"Step 14: {TICKER_TO_ANALYZE} Downtrend Episodes (>{DRAWDOWN_THRESHOLD:.0%} drawdown)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step14_downtrend_episodes_price.png", dpi=150)
    plt.show()

    # Plot histogram of trading-day durations
    plt.figure(figsize=(10, 6))
    plt.hist(episodes_df["trading_days"], bins=min(15, len(episodes_df)), edgecolor="black")
    plt.title("Step 14: Distribution of Downtrend Episode Durations")
    plt.xlabel("Trading Days")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step14_downtrend_duration_hist.png", dpi=150)
    plt.show()

    print("Saved plots:")
    print("- step14_downtrend_episodes_price.png")
    print("- step14_downtrend_duration_hist.png")


if __name__ == "__main__":
    main()