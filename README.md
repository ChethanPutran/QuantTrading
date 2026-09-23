# QuantTrading

**Adaptive Probabilistic Trading Intelligence System**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research--experimental-orange)]()

> **⚠️ Research / Experimental System**  
> This project is not intended for unattended live trading. Extensive validation, risk controls, and paper trading are required before any production deployment.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Core Design Principles](#core-design-principles)
- [Data Pipeline](#data-pipeline)
- [Machine Learning Components](#machine-learning-components)
- [Trading Engine](#trading-engine)
- [Metrics & Evaluation](#metrics--evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

QuantTrading is a real-time, online-learning research platform that combines **market microstructure analysis**, **probabilistic regime detection**, **pattern memory**, **hidden-state learning**, **machine-learning ensembles**, and **Model Predictive Control (MPC)** to generate adaptive trading decisions.

Unlike conventional offline machine-learning systems that train once on historical data, QuantTrading replays historical market data as if it were arriving live. It makes sequential predictions, observes outcomes, and continuously updates its knowledge base. The system maintains a long-term memory of previously observed market patterns through a Pattern Database, enabling it to recognise recurring regimes and adapt to structural market changes over time.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Online Learning** | Models update incrementally after every completed trade; no offline retraining required. |
| **Probabilistic Regimes** | Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) provide soft regime probabilities and transition dynamics. |
| **Pattern Memory** | A Pattern DB stores feature representations, regime probabilities, hidden variables, model predictions, and historical rewards. Unknown patterns are created and branched automatically. |
| **Multi-Model Ensemble** | Decision Trees, Gradient Boosting, SVM, LSTM, and regression models are combined with pattern and regime signals. |
| **Model Predictive Control** | MPC forecasts the next five price steps and optimises a reward/risk/cost objective. |
| **Noise Filtering** | A Kalman filter estimates a cleaner latent price state before regime and prediction stages. |
| **Multi-Horizon Support** | Independent model pipelines for 3, 5, 10, 20, 30-minute, and 1-day horizons. |
| **Event-Driven Async Architecture** | Non-blocking collectors, queues, and processing stages keep the real-time path responsive. |
| **Historical Replay** | Deterministic replay engine with configurable speed, simulated latency, and online model updates. |
| **Comprehensive Metrics** | Classification, regression, and trading metrics (PnL, Sharpe, drawdown, expectancy) are tracked continuously. |

---

## Architecture

The system follows a layered, event-driven pipeline. Each stage enriches the market state before passing it to the next:

```
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
│  Feature Engineering │
│  OHLCV · Bid/Ask     │
│  Spread · Depth      │
│  Imbalance · Options │
│  Technical Indicators│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Noise Filtering    │
│   Kalman Filter      │
└──────────┬───────────┘
           │
           ▼
┌───────────────────────────────────┐
│    Probabilistic State            │
│    GMM → Regime Probabilities     │
│    HMM → Regime Transitions       │
└────────────────┬──────────────────┘
           │
           ▼
┌───────────────────────────────────┐
│    Pattern Memory                 │
│    Pattern DB                     │
│    Similarity Search              │
│    Unknown Pattern Creation       │
│    Pattern Branching              │
└────────────────┬──────────────────┘
           │
           ▼
┌───────────────────────────────────┐
│    Hidden State Layer             │
│    Hidden Variables               │
│    Hidden Regime State            │
│    Pattern-specific Latent State  │
└────────────────┬──────────────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
┌────────┐ ┌──────────┐ ┌────────┐
│Decision│ │Gradient  │ │  SVM   │
│  Tree  │ │Boosting  │ │        │
└────────┘ └──────────┘ └────────┘
     │          │           │
     └──────────┼───────────┘
                ▼
     ┌────────────────────┐
     │   LSTM Models      │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │  Price Regression  │
     │  Future Estimates  │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │        MPC         │
     │  Predict t+1…t+5   │
     │  Optimise Reward/  │
     │  Risk/Cost         │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │  Ensemble Decision │
     │  BUY / SELL / HOLD │
     │  Confidence        │
     │  Expected Reward   │
     │  Expected Risk     │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │    Risk Engine     │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │ Execution Simulator│
     │    / Broker        │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │   Trade Outcome    │
     │  Profit / Loss     │
     │  Prediction Error  │
     │  Trajectory Error  │
     └─────────┬──────────┘
               ▼
     ┌────────────────────┐
     │  Online Learning   │
     │  Model Updates     │
     │  Pattern Updates   │
     │  Hidden Var Updates│
     │  GMM / HMM Updates │
     │  Ensemble Weights  │
     └─────────┬──────────┘
               │
               └────────────► Pattern DB
```

---

## Project Structure

```
QuantTrading/
├── config/
│   ├── config.yaml          # Primary system configuration
│   ├── constraint.yaml      # Risk and position constraints
│   └── logging.yaml         # Logging configuration
├── data/
│   └── raw/                 # Historical OHLCV datasets (NSE symbols)
├── notebooks/               # Jupyter notebooks for research and experiments
├── results/                 # Model artefacts, trade logs, analytics outputs
│   ├── LSTM_with_attention.csv
│   └── complete_supervised_trading_smoke/
├── src/                     # Source package (application logic)
├── main.py                  # CLI entry point
├── pyproject.toml           # Project metadata and dependencies
├── setup.sh                 # Environment bootstrap script
├── Makefile                 # Common development tasks
├── Dockerfile               # Container image definition
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`
- Docker (optional, for containerised deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/ChethanPutran/QuantTrading.git
cd QuantTrading

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Or use uv
uv sync
```

### Docker

```bash
docker build -t quanttrading .
docker run --rm -it quanttrading --steps 500
```

### Quick Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

---

## Usage

The system is driven through `main.py`, which supports both synthetic streaming and CSV-based historical replay.

### Synthetic Streaming

```bash
python main.py --symbol ^NSEI --steps 1000 --seed 42
```

### Historical Replay from CSV

```bash
python main.py --csv-path data/raw/ADANIPOWER.NS_data.csv --symbol ADANIPOWER.NS
```

### Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | `^NSEI` | Market symbol to stream or replay. |
| `--csv-path` | `None` | Path to a CSV file for historical replay. |
| `--steps` | `500` | Number of synthetic ticks to simulate. |
| `--delay-per-tick` | `0.0` | Delay in seconds between replay ticks. |
| `--seed` | `42` | Deterministic seed for synthetic replay. |
| `--state-store` | `results/state_store` | Directory for analytics outputs. |
| `--replay-store` | `results/replay_store` | Directory for replay logs. |
| `--redis-url` | `None` | Optional Redis URL for live state publication. |
| `--log-level` | `INFO` | Logging level. |

### Example Output

The system prints a JSON report containing performance metrics and output paths:

```json
{
  "report": {
    "trades": 12,
    "win_rate": 0.583,
    "pnl": 0.041,
    "sharpe": 1.24
  },
  "outputs": {
    "state_store": "results/state_store",
    "replay_store": "results/replay_store"
  }
}
```

---

## Configuration

Configuration is split across YAML files in `config/`:

- **`config.yaml`** – Core system parameters (symbols, horizons, model hyperparameters, ensemble weights).
- **`constraint.yaml`** – Position limits, max drawdown, risk thresholds.
- **`logging.yaml`** – Log levels, formatters, and handlers.

Runtime settings can be overridden via CLI arguments. For programmatic use, `main.py` constructs `AppSettings` from `ReplaySettings` and `RuntimeSettings` objects.

---

## Core Design Principles

### Online, Not Offline

The system never uses future information when making a prediction. Historical data is replayed event-by-event, and the model updates only after the outcome of each trade is observed:

```
Historical Event → Feature State → Prediction → Trade
       → Future Market Data → Outcome → Update → Next Event
```

### Independent Horizons

Each supported horizon (3m, 5m, 10m, 20m, 30m, 1d) maintains its own features, models, pattern memory, predictions, metrics, and trade outcomes. A shared representation may be introduced later, but horizons are isolated by default.

### Single Active Position

The trading state machine enforces strict position rules:

```
FLAT ──BUY──► LONG ──SELL──► FLAT
```

Invalid transitions (BUY → BUY, SELL → SELL, SELL while FLAT) are rejected. Only one position may be active at a time in the initial implementation.

---

## Data Pipeline

### Feature Engineering

Raw events are converted into a unified `StateVector` comprising:

- **Technical indicators:** RSI, MACD, EMA, SMA, ATR, VWAP, Bollinger Bands, momentum, rolling volatility, z-score.
- **Microstructure:** bid/ask spread, relative spread, bid/ask imbalance, order-flow imbalance, depth imbalance, liquidity pressure, trade intensity, volume imbalance, price impact, short-term volatility.
- **Options:** implied volatility, IV change, IV skew, open-interest change, call/put imbalance, strike concentration, gamma-related exposure.

### Noise Filtering

A Kalman filter estimates a cleaner latent price/state from the observed market state. The filtered state is passed into the regime and prediction systems.

### Data Sources

The initial implementation uses market data only:

- **Market:** OHLCV, bid, ask, last trade, spread, order book, maximum depth, bid depth, ask depth, order imbalance, trade volume, trade intensity, liquidity.
- **Options:** option OHLCV, bid, ask, last trade, open interest, implied volatility, strike, expiry, call/put information, call/put imbalance, IV skew, gamma-related features, strike concentration.

Future external data sources (news, economic data, interest rates, commodities, sentiment, alternative data) are designed to be pluggable as additional feature sources without redesigning the architecture.

---

## Machine Learning Components

| Component | Role |
|-----------|------|
| **GMM** | Identifies probabilistic market regimes (e.g., Regime 1: 0.10, Regime 2: 0.72, Regime 3: 0.13, Regime 4: 0.05). Supports online/incremental adaptation. |
| **HMM** | Models temporal regime transitions (Calm → Trending → High Volatility → Mean Reversion). Outputs hidden regime, transition probabilities, and state probabilities. |
| **Pattern DB** | Long-term memory storing pattern ID, parent ID, feature representation, regime probabilities, hidden variables, model predictions, confidence, historical reward, win rate, trade count, trajectory statistics, timestamps, and child patterns. |
| **Hidden State Layer** | Estimates hidden variables, hidden regime state, and pattern-specific latent state associated with the current pattern. |
| **Decision Tree / Gradient Boosting / SVM** | Classical supervised models for directional and regression tasks. |
| **LSTM** | Sequence model for temporal dependencies and price trajectory prediction. |
| **Price Regression** | Produces future price estimates from the ensemble of models. |
| **MPC** | Predicts the next five price steps and optimises a reward/risk/cost objective. |
| **Ensemble** | Combines pattern probability, GMM, HMM, tree models, boosting, SVM, LSTM, regression, hidden state, and MPC into a final action with confidence, expected price, expected reward, and expected risk. |

### Online Adaptation

After each completed trade:

- **Successful prediction:** increase pattern confidence, pattern reward, model contribution, and hidden-state association.
- **Failed prediction:** decrease pattern confidence and model contribution; potentially create a child pattern representing the newly discovered scenario.

---

## Trading Engine

### Decision Flow

1. Build the current feature state.
2. Filter market noise (Kalman).
3. Estimate probabilistic regimes (GMM/HMM).
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

### Storage

| Store | Purpose |
|-------|---------|
| **Redis** | Current state, active position, fast pattern access, runtime state. |
| **DuckDB** | Analytics, experiments, querying historical results. |
| **Parquet** | Raw historical events, replay datasets, feature datasets, experiment output. |

### Async Architecture

Collectors feed async queues, which drive the feature processor, state engine, prediction engine, decision engine, and execution layer. The architecture avoids blocking the real-time path.

---

## Metrics & Evaluation

### Classification

Accuracy, precision, recall, F1, true positives, false positives, true negatives, false negatives.

### Regression

RMSE, MAE, directional accuracy, trajectory error.

### Trading

PnL, win rate, profit factor, Sharpe ratio, maximum drawdown, expectancy.

### Visualisation

The system generates:

- Accuracy vs time
- RMSE vs time
- Confidence vs time
- F1 vs time
- TP/FP/TN/FN
- Predicted vs actual price
- MPC trajectories
- Regime probabilities
- Hidden-state evolution
- Pattern confidence
- Cumulative PnL
- Drawdown
- Trade entry/exit points

---

## Development

### Running Tests

```bash
pytest
```

### Linting and Formatting

```bash
ruff check .
ruff format .
```

### Make Targets

```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Run linters
make run        # Run synthetic simulation
```

### Building the Container

```bash
docker build -t quanttrading .
```

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch.
2. Ensure all tests pass and linting is clean.
3. Submit a pull request with a clear description of the change and its motivation.
4. For architectural changes, open an issue first to discuss the design.

---

## License

This project is provided for research and educational purposes. See the repository for license details.

---

## Acknowledgements

Built with Python, NumPy, pandas, scikit-learn, and related open-source libraries. Market data in `data/raw/` covers NSE-listed symbols for research replay.

---
