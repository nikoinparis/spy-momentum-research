# Project 2: SPY Momentum Research System

## Overview

This project explores whether a systematic momentum-based trading strategy can be built and improved for **SPY** using historical price data, risk management, and validation techniques.

The goal was not just to create a profitable backtest, but to build a **research pipeline** that answers questions such as:

- What kind of momentum signal works best on SPY?
- Does volatility targeting improve the strategy?
- Should the strategy be long/short or long-only?
- Can a defensive filter improve risk-adjusted performance?
- Does the strategy hold up under validation and Monte Carlo simulation?

This project gradually evolved from a simple baseline momentum strategy into a stronger and more interpretable final model.

---

## Core Idea

The project started with a basic **time-series momentum** framework:

- if momentum is positive, take a position
- if momentum is negative, either short or stay out
- evaluate performance using returns, Sharpe ratio, drawdown, and equity curves

From there, the strategy was improved step by step through:

- signal tuning
- volatility targeting
- threshold testing
- robustness sweeps
- benchmark comparison
- train/test validation
- forward validation on newly downloaded data
- Monte Carlo simulation

---

## Data

The project uses **daily SPY price data** downloaded with `yfinance`.

Main processing steps:
- load historical adjusted close prices
- compute daily log returns
- generate momentum signals from rolling lookback windows
- backtest strategy returns with transaction costs
- apply volatility targeting for risk control

---

## Research Process

## 1. Baseline Momentum Strategy

The starting point was a simple momentum rule using a fixed lookback window.

Initial tests showed that:
- short lookbacks like 5 and 10 days performed poorly
- 20-day momentum was better, but still not ideal
- the strategy needed better signal design

This showed that raw momentum alone was not enough.

---

## 2. Volatility Targeting

A volatility-targeting overlay was added to scale exposure based on recent realized volatility.

This helped answer:
- whether risk-adjusted performance improves when the strategy sizes down in high-volatility periods
- whether a strong signal becomes more stable with dynamic risk scaling

Volatility targeting improved some versions of the strategy significantly, especially after the signal itself was improved.

---

## 3. Threshold Filter Testing

A threshold filter was tested to avoid trading weak momentum signals.

The idea was:
- only trade if the momentum signal is strong enough
- stay flat when the signal is too small

This helped test whether weaker signals were mostly noise.

Result:
- large thresholds generally hurt performance
- a very small threshold could help in some cases
- strong filtering was not beneficial for SPY in this framework

---

## 4. Lookback Sweep

A sweep across different momentum lookback windows was one of the most important experiments.

Lookbacks tested included:
- 5
- 10
- 20
- 40
- 60
- 120

### Main finding
**60-day momentum** was the strongest lookback among the tested options.

This suggested that SPY responds better to **medium-term trend behavior** than to very short-term momentum.

---

## 5. Benchmark Comparison

The strategy was compared against:

- an equal-weight benchmark / passive exposure
- earlier baseline momentum versions
- improved versions of the same strategy family

This was important because it showed whether strategy improvements were meaningful relative to simpler alternatives.

### Main finding
The improved strategies beat the original baseline by a wide margin, but passive exposure remained a strong benchmark.

This reinforced the idea that:
- SPY has a strong upward drift
- strategies should respect that bias
- blindly shorting negative momentum may reduce performance

---

## 6. Train/Test Validation

A train/test split was used to check whether the strategy remained effective outside the earlier sample.

This step helped answer:
- whether the strategy only looked good in one historical region
- whether the logic still behaved reasonably in a later period

### Main finding
The later period did not show a collapse.  
This suggested that the final strategy design was more robust than the original baseline.

---

## 7. Forward Validation on New Data

The project then extended the historical data and tested the frozen strategy on newly downloaded unseen data.

This was used to simulate a more realistic forward-style validation.

### Main finding
The short unseen window was mildly negative, but not catastrophic.  
This was interpreted carefully, since short horizons can be noisy.

---

## 8. Monte Carlo Simulation

Bootstrap Monte Carlo simulation was used to estimate the distribution of possible future outcomes based on the historical return stream of the strategy.

This answered questions like:
- What is the probability of profit over a given horizon?
- What kind of drawdowns are plausible?
- Are short losing stretches normal?

Two types of horizons were especially useful:
- short horizon (around 50 trading days)
- one-year horizon (252 trading days)

---

## 9. Long-Only vs Long-Short

A major turning point in the project was testing whether SPY should be treated as a symmetric long/short market.

### Main finding
**Long-only momentum clearly outperformed long-short momentum.**

This was a strong result and matched the economic intuition that SPY has a long-run upward bias.

That meant:
- the long side carried the edge
- the short side often hurt returns
- a better strategy should respect SPY’s structural upward drift

---

## 10. Short-Only Analysis

Short-only momentum and short-lookback sweeps were tested separately to see whether a better short logic could be found.

### Main finding
Short-only momentum performed poorly across all tested lookbacks.

The “best” short lookback was only the **least bad**, not genuinely strong.

This suggested that:
- shorting SPY is difficult
- short-side momentum is not a major source of alpha in this framework
- going flat in weak regimes may be better than trying to short every downturn

---

## 11. Downtrend Episode Analysis

The project also analyzed historical SPY downtrend episodes by measuring:
- duration
- time to trough
- maximum drawdown
- recovery behavior

### Main finding
SPY drawdowns were often:
- short
- recoverable
- not ideal for simple momentum-based shorting

This helped explain why short-only momentum underperformed.

---

## 12. Defensive Filter

A **200-day moving average filter** was then added to the long-only momentum strategy.

New rule:
- only allow long positions when SPY is above its 200-day moving average
- otherwise stay flat

### Main finding
This was the strongest improvement in the whole project.

The filter:
- increased final equity
- increased annual return
- reduced annual volatility
- improved Sharpe ratio
- reduced maximum drawdown

This showed that for SPY, a **defensive long-only trend-following approach** works better than a symmetric long/short framework.

---

## 13. Mean-Reversion Overlay

A mean-reversion overlay was tested on top of the defensive long-only strategy.

The idea was to buy oversold dips in an otherwise bullish regime.

### Main finding
The overlay was reasonable, but it did **not** improve the final strategy enough to beat the simpler filtered long-only version.

This meant the stronger final model was still the simpler one.

---

## Final Strategy

The final champion strategy from Project 2 is:

- **Long-only momentum**
- **60-day lookback**
- **0.001 threshold**
- **200-day moving average defensive filter**
- **Volatility targeting**
- **Target volatility = 10%**
- **Volatility window = 40 days**

---

## Final Findings

The main things this project found are:

### 1. SPY momentum works better as long-only than long/short
The long side is the main source of edge.  
Shorting SPY in a simple momentum framework generally reduced performance.

### 2. Medium-term momentum is better than short-term momentum
Among the tested windows, **60-day momentum** was the strongest.

### 3. Volatility targeting helps when the underlying signal is good
Risk scaling improved the stronger versions of the strategy more than the weaker ones.

### 4. Defensive filtering is more useful than symmetric shorting
The **200-day moving average filter** improved both return and risk-adjusted performance.

### 5. Simplicity won over extra overlays
A simple oversold mean-reversion overlay did not improve the final model enough, so the filtered long-only trend strategy remained the winner.

### 6. The final strategy looks much stronger than the original baseline
Compared with the early baseline momentum strategies, the final strategy was:
- more profitable
- more stable
- more interpretable
- better aligned with SPY’s long-run behavior

---

## What This Project Demonstrates

This project is not just one backtest. It is a **research workflow**.

It demonstrates:
- how to start from a weak baseline
- how to improve a strategy step by step
- how to validate ideas rather than guess
- how to compare models honestly
- how to use Monte Carlo and forward testing to understand uncertainty

---

## Possible Next Steps

Future improvements could include:
- walk-forward optimization
- alternative defensive filters
- cross-asset testing beyond SPY
- different long-only entry/exit rules
- transaction cost sensitivity analysis
- regime-based allocation across multiple strategies

---

## Repository Structure

Example structure:

```text
src/
  data_loader.py
  strategies.py
  backtester.py
  metrics.py
  config.py

scripts/
  project2_*.py

data/
  raw/
```


⸻

Summary

Project 2 began as a basic momentum experiment and evolved into a more robust SPY long-only trend-following system with defensive filtering and volatility targeting.

The final result suggests that for SPY:
	•	respecting long-run upward drift matters
	•	long-only trend-following works better than symmetric momentum
	•	going flat in weak regimes is more effective than forcing shorts
	•	a simple, disciplined structure can outperform more complicated variations
