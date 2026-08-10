# Adaptive Probabilistic Trading Intelligence System

A real-time, online-learning trading research platform that combines **market microstructure, probabilistic regime detection, pattern memory, hidden-state learning, machine-learning ensembles, and Model Predictive Control (MPC)** to generate adaptive trading decisions.

> **Status:** Research / experimental system. Not intended for unattended live trading without extensive validation, risk controls, and paper-trading.

---

## 1. Overview

The system is designed around one central idea:

> Instead of training a single static model on historical data, replay historical market data as if it were arriving live, make predictions sequentially, observe the outcome, and continuously update the system's knowledge.

The system maintains a memory of previously observed market patterns.

When a new market state arrives:

1. Build the current feature state.
2. Filter market noise.
3. Estimate probabilistic regimes.
4. Search the Pattern DB for previously observed states.
5. Identify the most probable known pattern.
6. If no suitable pattern exists, create an `UNKNOWN` pattern.
7. Estimate hidden variables associated with the pattern.
8. Run multiple prediction models.
9. Predict the next five price steps using MPC.
10. Combine all predictions using an ensemble.
11. Generate `BUY`, `SELL`, or `HOLD`.
12. Enforce position constraints.
13. Evaluate the trade after the required horizon.
14. Update model weights, pattern confidence, hidden variables, GMM/HMM state, and pattern branches.

---

# 2. System Architecture

```text
                         ┌──────────────────────┐
                         │   Market Data Feed   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Event Processing   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         │                      │
                         │ OHLCV                │
                         │ Bid / Ask            │
                         │ Spread               │
                         │ Depth                │
                         │ Imbalance            │
                         │ Options              │
                         │ Technical Indicators │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Noise Filtering    │
                         │    Kalman Filter     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌───────────────────────────────────┐
                  │       Probabilistic State         │
                  │                                   │
                  │ GMM → Regime Probabilities       │
                  │ HMM → Regime Transitions         │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │         Pattern Memory            │
                  │                                   │
                  │ Pattern DB                         │
                  │ Similarity Search                  │
                  │ GMM Pattern Probability            │
                  │ Unknown Pattern Creation          │
                  │ Pattern Branching                  │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │        Hidden State Layer         │
                  │                                   │
                  │ Hidden Variables                  │
                  │ Hidden Regime State               │
                  │ Pattern-specific latent state    │
                  └────────────────┬──────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          Decision Tree      Gradient Boost      SVM
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                              LSTM Models
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │        Price Regression           │
                  │                                   │
                  │ Future price estimates            │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │             MPC                   │
                  │                                   │
                  │ Predict next 5 price steps        │
                  │ Optimize reward / risk / cost     │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │       Ensemble Decision           │
                  │                                   │
                  │ BUY / SELL / HOLD                 │
                  │ Confidence                        │
                  │ Expected reward                    │
                  │ Expected risk                      │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │          Risk Engine              │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │       Execution Simulator         │
                  │          / Broker                 │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │       Trade Outcome               │
                  │                                   │
                  │ Profit / Loss                      │
                  │ Prediction Error                   │
                  │ Trajectory Error                   │
                  └────────────────┬──────────────────┘
                                   │
                                   ▼
                  ┌───────────────────────────────────┐
                  │       Online Learning              │
                  │                                   │
                  │ Model Updates                      │
                  │ Pattern Updates                    │
                  │ Hidden Variable Updates            │
                  │ GMM / HMM Updates                  │
                  │ Ensemble Weight Updates            │
                  └────────────────┬──────────────────┘
                                   │
                                   └──────────────► Pattern DB
```

---

# 3. Main Design Principle

The system is **online**, not conventional offline ML.

Historical training should behave like:

```text
Historical Event
      ↓
Feature State
      ↓
Prediction
      ↓
Trade
      ↓
Future Market Data
      ↓
Outcome
      ↓
Update
      ↓
Next Event
```

The model must never use future information when making a prediction.

---

# 4. Supported Training Horizons

The system must independently support:

```text
3 minutes
5 minutes
10 minutes
20 minutes
30 minutes
1 day
```

The initial training dataset should contain approximately:

```text
6 months of historical data
```

Each timeframe should maintain its own:

* features
* models
* pattern memory
* predictions
* metrics
* trade outcomes

A shared representation may optionally be introduced later.

---

# 5. Initial Data Sources

The first implementation should use **market data only**.

Future external data should be pluggable without redesigning the architecture.

## Market Data

Collect:

* OHLCV
* Bid
* Ask
* Last Trade
* Spread
* Order Book
* Maximum Depth
* Bid Depth
* Ask Depth
* Order Imbalance
* Trade Volume
* Trade Intensity
* Liquidity

## Options

Collect:

* Option OHLCV
* Bid
* Ask
* Last Trade
* Open Interest
* Implied Volatility
* Strike
* Expiry
* Call/Put information
* Call/Put imbalance
* IV skew
* Gamma-related features
* Strike concentration

---

# 6. Future Data Sources

The architecture should later support:

```text
News
Economic Data
Interest Rates
Oil
Natural Gas
Gold
Silver
Forex
Bonds
Indices
VIX
Social Sentiment
Alternative Data
Earnings
Economic Calendar
```

These should become additional feature sources rather than being tightly coupled to the initial market-data implementation.

---

# 7. Feature Engineering

The feature pipeline converts raw events into a unified `StateVector`.

## Technical Indicators

Initially implement:

* RSI
* MACD
* EMA
* SMA
* ATR
* VWAP
* Bollinger Bands
* Momentum
* Rolling volatility
* Z-score

## Microstructure

Implement:

* Bid/Ask spread
* Relative spread
* Bid/Ask imbalance
* Order-flow imbalance
* Depth imbalance
* Liquidity pressure
* Trade intensity
* Volume imbalance
* Price impact
* Short-term volatility

## Options

Implement:

* IV
* IV change
* IV skew
* Open-interest change
* Call/put imbalance
* Strike concentration
* Gamma-related exposure

---

# 8. Noise Filtering

Use a Kalman filter to estimate a cleaner latent price/state.

Conceptually:

```text
Observed Market State
        ↓
     Kalman
        ↓
Estimated Latent State
```

The filtered state is passed into the regime and prediction systems.

---

# 9. GMM Regime Detection

Use Gaussian Mixture Models to identify probabilistic market states.

Instead of saying:

```text
Current regime = 2
```

the system should maintain:

```text
Regime 1: 0.10
Regime 2: 0.72
Regime 3: 0.13
Regime 4: 0.05
```

This probability distribution becomes part of the feature state.

The GMM should support online/incremental adaptation where practical.

---

# 10. HMM Regime Transitions

Use an HMM to model temporal regime transitions.

For example:

```text
Calm
  ↓
Trending
  ↓
High Volatility
  ↓
Mean Reversion
```

The HMM maintains:

* hidden regime
* transition probabilities
* state probabilities

The HMM output becomes another input to the ensemble.

---

# 11. Pattern Database

The Pattern DB is the system's long-term memory.

A pattern should contain approximately:

```text
Pattern ID
Parent Pattern ID
Feature Representation
Regime Probabilities
Hidden Variables
Model Predictions
Prediction Confidence
Historical Reward
Win Rate
Trade Count
Trajectory Statistics
Creation Time
Last Updated
Child Patterns
```

---

# 12. Pattern Identification

For every new state:

```text
Current State
     ↓
Encode
     ↓
Pattern Search
     ↓
Known Pattern?
```

### If known:

```text
Retrieve pattern
      ↓
Estimate probability
      ↓
Use as prior scenario
```

### If unknown:

```text
Create UNKNOWN pattern
      ↓
Initialize hidden variables
      ↓
Store pattern
```

The system should identify which known pattern most closely generated the current state.

---

# 13. Hidden Variables

Each pattern can contain latent variables representing factors not directly observed.

For example:

```text
Hidden Variable 1 → liquidity pressure
Hidden Variable 2 → institutional activity
Hidden Variable 3 → latent volatility
Hidden Variable 4 → hidden momentum
```

Initially these are not assumed to have a known meaning.

Their associations should emerge through learning.

The system should eventually be able to analyze relationships such as:

```text
Hidden Variable X
       ↓
correlates with
       ↓
high imbalance + high volatility
```

This allows hidden variables to become interpretable over time.

---

# 14. Hidden Variable Updates

After a trade finishes:

```text
Prediction
   ↓
Actual Outcome
   ↓
Prediction Error
   ↓
PnL
   ↓
Hidden State Update
```

Update the hidden variables associated with the pattern that generated the prediction.

The objective is to learn:

> Which latent conditions explain why a particular pattern produced the observed outcome?

---

# 15. Prediction Models

Run multiple models in parallel.

## Classification

Predict:

```text
BUY
SELL
HOLD
```

Models:

* Decision Tree
* Gradient Boosting
* SVM
* LSTM

## Regression

Predict future prices using:

* Tree-based regression
* Gradient Boosting Regression
* LSTM Regression

Each model should produce:

```text
Prediction
Confidence
```

where appropriate.

---

# 16. MPC

Use Model Predictive Control to estimate the next five price steps.

Example:

```text
t+1
t+2
t+3
t+4
t+5
```

The MPC should consider:

* predicted price
* volatility
* hidden state
* regime
* uncertainty
* transaction cost
* risk

The trajectory becomes another input to the final decision engine.

---

# 17. Ensemble

The final action should not depend on a single model.

Combine:

```text
Pattern Probability
GMM
HMM
Decision Tree
Gradient Boosting
SVM
LSTM
Regression Models
Hidden State
MPC
```

The ensemble produces:

```text
Action
Confidence
Expected Price
Expected Reward
Expected Risk
```

---

# 18. Trading State Machine

Trading must obey:

```text
FLAT
  │
  └── BUY ──► LONG
                │
                └── SELL ──► FLAT
```

Never allow:

```text
BUY → BUY
SELL → SELL
SELL while FLAT
```

Only one position may be active at a time in the initial implementation.

---

# 19. Online Adaptation

After each completed trade:

### Successful prediction

Increase:

* pattern confidence
* pattern reward
* model contribution
* hidden-state association

### Failed prediction

Decrease:

* pattern confidence
* model contribution

and potentially:

```text
Parent Pattern
      ↓
New Child Pattern
```

The child pattern represents the newly discovered scenario.

---

# 20. Metrics

Track classification:

```text
Accuracy
Precision
Recall
F1
TP
FP
TN
FN
```

Regression:

```text
RMSE
MAE
Directional Accuracy
Trajectory Error
```

Trading:

```text
PnL
Win Rate
Profit Factor
Sharpe Ratio
Maximum Drawdown
Expectancy
```

---

# 21. Visualization

Generate:

* Accuracy vs time
* RMSE vs time
* Confidence vs time
* F1 vs time
* TP/FP/TN/FN
* Predicted vs actual price
* MPC trajectories
* Regime probabilities
* Hidden-state evolution
* Pattern confidence
* Cumulative PnL
* Drawdown
* Trade entry/exit points

---

# 22. Historical Replay

The replay engine should make historical data behave like live data.

```text
Historical Data
      ↓
Replay Engine
      ↓
Real-time Event Stream
      ↓
Trading System
```

It should support:

* configurable replay speed
* deterministic replay
* event timestamps
* simulated latency
* simulated execution
* online model updates

---

# 23. Storage

Use different storage systems for different purposes.

### Redis

For:

* current state
* active position
* fast pattern access
* runtime state

### DuckDB

For:

* analytics
* experiments
* querying historical results

### Parquet

For:

* raw historical events
* replay datasets
* feature datasets
* experiment output

---

# 24. Async Architecture

Use an asynchronous event-driven architecture:

```text
Collectors
    ↓
Async Queues
    ↓
Feature Processor
    ↓
State Engine
    ↓
Prediction Engine
    ↓
Decision Engine
    ↓
Execution
```

The architecture should avoid blocking the real-time path.

---

# 25. Recommended Project Structure

```text
trading-system/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── system.yaml
│   ├── models.yaml
│   ├── features.yaml
│   └── risk.yaml
│
├── src/
│   ├── core/
│   ├── data/
│   │   ├── market/
│   │   └── options/
│   │
│   ├── features/
│   ├── filtering/
│   ├── regime/
│   │   ├── gmm/
│   │   └── hmm/
│   │
│   ├── memory/
│   │   └── pattern_db/
│   │
│   ├── hidden_state/
│   ├── models/
│   │   ├── decision_tree/
│   │   ├── gradient_boosting/
│   │   ├── svm/
│   │   └── lstm/
│   │
│   ├── regression/
│   ├── control/
│   │   └── mpc/
│   │
│   ├── ensemble/
│   ├── risk/
│   ├── execution/
│   ├── learning/
│   ├── replay/
│   ├── storage/
│   ├── monitoring/
│   └── visualization/
│
├── tests/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── replay/
│
├── models/
│
└── experiments/
```

---

# 26. Development Phases

## Phase 1 — Market Data

Implement:

* market collector
* option collector
* event schemas
* storage
* replay

## Phase 2 — Features

Implement:

* technical indicators
* microstructure features
* option features

## Phase 3 — State Estimation

Implement:

* Kalman
* GMM
* HMM

## Phase 4 — Pattern Memory

Implement:

* Pattern DB
* similarity search
* pattern probability
* unknown patterns
* branching

## Phase 5 — ML

Implement:

* Decision Tree
* Gradient Boosting
* SVM
* LSTM
* regression models

## Phase 6 — MPC

Implement:

* five-step trajectory prediction
* risk-aware optimization

## Phase 7 — Ensemble

Combine all model outputs.

## Phase 8 — Online Learning

Implement:

* reward updates
* model adaptation
* hidden-state updates
* pattern branching

## Phase 9 — Evaluation

Implement:

* metrics
* charts
* experiments
* six-month replay

## Phase 10 — Paper Trading

Only after replay validation, run the complete system against a paper/simulated execution environment.

---

# 27. Important Design Principle

The system should **not be treated as one giant ML model**.

It is a collection of cooperating components:

```text
State Estimation
       +
Regime Detection
       +
Pattern Memory
       +
Hidden State
       +
Multiple Predictors
       +
MPC
       +
Risk
       +
Online Learning
```

The key research component is the feedback loop:

```text
Observe
   ↓
Recognize Pattern
   ↓
Predict
   ↓
Trade
   ↓
Observe Outcome
   ↓
Measure Error
   ↓
Update Hidden State
   ↓
Update Pattern
   ↓
Update Models
   ↓
Learn
   ↓
Observe Again
```

This feedback loop is what makes the system **adaptive rather than static**.
