"""
Supervised Model Trading Simulation
===================================
Uses trained supervised models (action classifier + profit regressor) to simulate trading.

Features:
- Loads pre-trained supervised models
- Uses action predictions for BUY/SELL/HOLD decisions
- Uses profit predictions for position sizing
- Tracks portfolio value, trades, and PnL
- Compares against unsupervised baseline models
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.market_data.option_chain import get_index_option_chain_history


@dataclass
class TickData:
    """Single tick simulation data."""

    date: str
    price: float
    regime: int
    prediction: float
    supervised_action: int
    supervised_confidence: float
    quantity: int
    cash: float
    position: float
    portfolio_value: float
    pnl: float


class SupervisedTradingSimulator:
    """Simulates trading using supervised models."""

    def __init__(
        self,
        symbol: str = "^NSEI",
        total_days: int = 210,
        simulate_days: int = 30,
        models_path: str | None = None,
    ):
        """
        Initialize simulator.

        Args:
            symbol: Stock symbol
            total_days: Total historical days to fetch
            simulate_days: Days to simulate on
            models_path: Path to supervised_models.joblib
        """
        self.symbol = symbol
        self.total_days = total_days
        self.simulate_days = simulate_days
        self.models_path = models_path

        self.historical_data: pd.DataFrame | None = None
        self.supervised_models: dict[str, Any] | None = None
        self.simulation_ticks: list[TickData] = []
        self.metrics: dict[str, Any] = {}

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data."""
        print(f"Fetching {self.total_days} days of historical data...")

        snapshots = get_index_option_chain_history(
            self.symbol,
            expiry="",
            days=self.total_days,
            interval="1d",
        )

        rows = []
        for snap in snapshots:
            rows.append(
                {
                    "date": pd.to_datetime(snap.get("date")),
                    "close": float(snap.get("underlying_price", np.nan)),
                }
            )

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        self.historical_data = df.dropna().reset_index(drop=True)
        print(f"✓ Loaded {len(self.historical_data)} trading days")
        return self.historical_data

    def load_supervised_models(self, models_path: str | None = None) -> dict[str, Any]:
        """
        Load pre-trained supervised models.

        Args:
            models_path: Path to supervised_models.joblib

        Returns:
            Loaded models dictionary
        """
        path = models_path or self.models_path
        if not path:
            raise ValueError("No models_path provided")

        print(f"Loading supervised models from {path}...")
        self.supervised_models = joblib.load(path)
        print("✓ Models loaded")
        return self.supervised_models

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features."""
        out = df.copy()
        close = out["close"].astype(float)

        out["ret_1"] = close.pct_change()
        out["ret_5"] = close.pct_change(5)
        out["momentum_10"] = close / close.shift(10) - 1.0

        out["ema_12"] = close.ewm(span=12, adjust=False).mean()
        out["ema_26"] = close.ewm(span=26, adjust=False).mean()
        out["macd"] = out["ema_12"] - out["ema_26"]

        out["vol_10"] = out["ret_1"].rolling(10).std()
        out["vol_30"] = out["ret_1"].rolling(30).std()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))

        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        out["bb_upper"] = sma + (std * 2)
        out["bb_lower"] = sma + (std * -2)
        out["bb_position"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

        return out.dropna().reset_index(drop=True)

    def simulate(self, initial_capital: float = 100000.0) -> list[TickData]:
        """
        Simulate trading using supervised models.

        Args:
            initial_capital: Starting portfolio value

        Returns:
            List of tick data from simulation
        """
        if self.historical_data is None:
            self.fetch_historical_data()
        if self.supervised_models is None:
            self.load_supervised_models()

        print("\nPreparing simulation data...")
        df = self._add_technical_features(self.historical_data)

        # Split into train and simulate periods
        train_size = len(df) - self.simulate_days
        df_sim = df.iloc[train_size:].reset_index(drop=True)

        # Extract feature columns
        feature_cols = [
            "ret_1",
            "ret_5",
            "momentum_10",
            "ema_12",
            "ema_26",
            "macd",
            "vol_10",
            "vol_30",
            "rsi_14",
            "bb_position",
        ]

        # Get models
        classifier = self.supervised_models["action_classifier"]
        regressor = self.supervised_models["profit_regressor"]
        scaler = self.supervised_models["feature_scaler"]

        print(f"Simulating {len(df_sim)} days of trading...")

        # Initialize portfolio
        cash = initial_capital
        position = 0.0
        trades = {"buy": 0, "sell": 0}
        ticks: list[TickData] = []

        for idx, row in df_sim.iterrows():
            date = str(row["date"])
            price = row["close"]

            # Extract and scale features
            X_features = np.array([[row.get(col, 0) for col in feature_cols]])
            X_scaled = scaler.transform(X_features)

            # Model predictions
            action = int(classifier.predict(X_scaled)[0])
            prob = np.max(classifier.predict_proba(X_scaled))
            expected_profit = float(regressor.predict(X_scaled)[0])

            # Simple strategy: position sizing based on profit expectation
            if action == 1 and expected_profit > 0.5:  # BUY signal
                # Buy 2-4% of portfolio per signal
                qty = (cash * 0.03) / price
                cost = qty * price
                if cash >= cost:
                    cash -= cost
                    position += qty
                    trades["buy"] += 1
                    action_taken = 1
                else:
                    action_taken = 0

            elif action == -1 and expected_profit > 0.5:  # SELL signal
                # Sell half of position
                if position > 0:
                    sell_qty = position / 2
                    cash += sell_qty * price
                    position -= sell_qty
                    trades["sell"] += 1
                    action_taken = -1
                else:
                    action_taken = 0
            else:
                action_taken = 0

            # Calculate portfolio value
            position_value = position * price
            portfolio_value = cash + position_value
            pnl = portfolio_value - initial_capital

            tick = TickData(
                date=date,
                price=price,
                regime=0,
                prediction=expected_profit,
                supervised_action=action,
                supervised_confidence=float(prob),
                quantity=int(qty) if action == 1 else 0,
                cash=cash,
                position=position,
                portfolio_value=portfolio_value,
                pnl=pnl,
            )
            ticks.append(tick)

        self.simulation_ticks = ticks
        self.metrics = {
            "symbol": self.symbol,
            "total_ticks": len(ticks),
            "initial_capital": initial_capital,
            "final_portfolio_value": ticks[-1].portfolio_value if ticks else initial_capital,
            "final_pnl": ticks[-1].pnl if ticks else 0.0,
            "buy_trades": trades["buy"],
            "sell_trades": trades["sell"],
            "mean_confidence": float(np.mean([t.supervised_confidence for t in ticks])),
            "max_drawdown": self._calculate_max_drawdown([t.portfolio_value for t in ticks]),
        }

        print(f"✓ Simulation complete!")
        print(f"  Final Portfolio Value: ${self.metrics['final_portfolio_value']:.2f}")
        print(f"  Final PnL: ${self.metrics['final_pnl']:.2f}")
        print(f"  Buy trades: {trades['buy']}, Sell trades: {trades['sell']}")

        return ticks

    def _calculate_max_drawdown(self, portfolio_values: list[float]) -> float:
        """Calculate maximum drawdown."""
        if not portfolio_values:
            return 0.0
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (cummax - np.array(portfolio_values)) / cummax
        return float(np.max(drawdown))

    def export_simulation(self, output_csv: str | None = None, output_json: str | None = None) -> dict[str, str]:
        """
        Export simulation results.

        Args:
            output_csv: Path to save tick-by-tick CSV
            output_json: Path to save summary JSON

        Returns:
            Dictionary with paths to exported files
        """
        paths = {}

        if output_csv:
            rows = []
            for tick in self.simulation_ticks:
                rows.append(
                    {
                        "date": tick.date,
                        "price": tick.price,
                        "action": tick.supervised_action,
                        "confidence": tick.supervised_confidence,
                        "expected_profit": tick.prediction,
                        "quantity": tick.quantity,
                        "position": tick.position,
                        "cash": tick.cash,
                        "portfolio_value": tick.portfolio_value,
                        "pnl": tick.pnl,
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
            print(f"✓ Exported simulation to {output_csv}")
            paths["csv"] = output_csv

        if output_json:
            with open(output_json, "w") as f:
                json.dump(self.metrics, f, indent=2)
            print(f"✓ Exported metrics to {output_json}")
            paths["json"] = output_json

        return paths


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Simulate trading using supervised models"
    )
    parser.add_argument("--symbol", default="^NSEI", help="Stock symbol")
    parser.add_argument(
        "--total-days",
        type=int,
        default=210,
        help="Total historical days",
    )
    parser.add_argument(
        "--simulate-days",
        type=int,
        default=30,
        help="Days to simulate",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Path to supervised_models.joblib",
    )
    parser.add_argument(
        "--output-csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--output-json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    simulator = SupervisedTradingSimulator(
        symbol=args.symbol,
        total_days=args.total_days,
        simulate_days=args.simulate_days,
        models_path=args.models,
    )

    simulator.fetch_historical_data()
    simulator.load_supervised_models()
    simulator.simulate()

    if args.output_csv or args.output_json:
        simulator.export_simulation(
            output_csv=args.output_csv,
            output_json=args.output_json,
        )

    print("\n" + "=" * 60)
    print(json.dumps(simulator.metrics, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
