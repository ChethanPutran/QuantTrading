"""
Trade Analysis Module
=====================
Analyzes historical trades to identify the most profitable trades per day
and generates supervised training data for model retraining.

This module:
1. Fetches historical price data for 6 months
2. Simulates trades at different price levels
3. Identifies top 10 most profitable trades per day
4. Generates supervised training data (features -> best_trades)
5. Creates labels for supervised learning
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.market_data.option_chain import get_index_option_chain_history


@dataclass
class Trade:
    """Represents a single trade with entry/exit points and profit."""
    date: str
    entry_price: float
    exit_price: float
    profit: float
    profit_pct: float
    trade_direction: int  # 1 for BUY, -1 for SELL
    holding_period: int  # days
    rank: int  # rank among all trades that day


@dataclass
class TradeOpportunity:
    """Supervised training sample: features -> best_action."""
    date: str
    close_price: float
    features: dict[str, float]  # Technical indicators
    best_action: int  # 1=BUY most profitable, -1=SELL most profitable, 0=HOLD
    expected_profit: float
    confidence: float  # Based on profit consistency


class TradeAnalyzer:
    """Analyzes historical trades and generates supervised training data."""

    def __init__(self, symbol: str = "^NSEI", lookback_days: int = 180):
        """
        Initialize trade analyzer.

        Args:
            symbol: Ticker symbol (default NIFTY50)
            lookback_days: Historical window size (default 6 months)
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.historical_data: pd.DataFrame | None = None
        self.daily_trades: dict[str, list[Trade]] = {}
        self.training_samples: list[TradeOpportunity] = []

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical OHLC data for the symbol."""
        print(f"Fetching {self.lookback_days} days of historical data for {self.symbol}...")

        snapshots = get_index_option_chain_history(
            self.symbol,
            expiry="",
            days=self.lookback_days,
            interval="1d",
        )

        print(f"✓ Retrieved {len(snapshots)} data points from market data API")
        print("Processing historical data...")

        rows = []
        for snap in snapshots:
            rows.append(
                {
                    "date": pd.to_datetime(snap.get("date")),
                    "close": float(snap.get("underlying_price", np.nan)),
                }
            )

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        # Add synthetic OHLC from close prices
        df["open"] = df["close"].shift(1)
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        df["volume"] = 1000000  # Placeholder

        self.historical_data = df.dropna().reset_index(drop=True)
        print(f"✓ Loaded {len(self.historical_data)} trading days")
        return self.historical_data

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        out = df.copy()
        close = out["close"].astype(float)

        # Momentum indicators
        out["ret_1"] = close.pct_change()
        out["ret_5"] = close.pct_change(5)
        out["momentum_10"] = close / close.shift(10) - 1.0

        # Trend indicators
        out["ema_12"] = close.ewm(span=12, adjust=False).mean()
        out["ema_26"] = close.ewm(span=26, adjust=False).mean()
        out["macd"] = out["ema_12"] - out["ema_26"]

        # Volatility
        out["vol_10"] = out["ret_1"].rolling(10).std()
        out["vol_30"] = out["ret_1"].rolling(30).std()

        # Momentum
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        out["bb_upper"] = sma + (std * 2)
        out["bb_lower"] = sma + (std * -2)
        out["bb_position"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

        return out.dropna().reset_index(drop=True)

    def _simulate_trades(self, df: pd.DataFrame, day_idx: int, holding_periods: list[int] = None) -> list[Trade]:
        """
        Simulate possible trades for a given day.

        Args:
            df: Historical data with technical features
            day_idx: Day index
            holding_periods: List of holding periods to test (default: 1-5 days)

        Returns:
            List of trades with their profits
        """
        if holding_periods is None:
            holding_periods = [1, 2, 3, 5]

        if day_idx >= len(df) - max(holding_periods):
            return []

        entry_date = df.iloc[day_idx]
        entry_price = entry_date["close"]
        entry_date_str = str(entry_date["date"])
        trades = []

        # Simulate BUY trades at different price levels
        for price_offset_pct in [-2, -1, 0, 1, 2]:  # -2% to +2%
            entry_at = entry_price * (1 + price_offset_pct / 100)

            for holding_period in holding_periods:
                exit_idx = day_idx + holding_period
                if exit_idx >= len(df):
                    continue

                exit_price = df.iloc[exit_idx]["close"]
                profit = exit_price - entry_at
                profit_pct = (profit / entry_at) * 100

                trades.append(
                    Trade(
                        date=entry_date_str,
                        entry_price=entry_at,
                        exit_price=exit_price,
                        profit=profit,
                        profit_pct=profit_pct,
                        trade_direction=1,  # BUY
                        holding_period=holding_period,
                        rank=0,
                    )
                )

        # Simulate SELL trades
        for price_offset_pct in [-2, -1, 0, 1, 2]:
            entry_at = entry_price * (1 + price_offset_pct / 100)

            for holding_period in holding_periods:
                exit_idx = day_idx + holding_period
                if exit_idx >= len(df):
                    continue

                exit_price = df.iloc[exit_idx]["close"]
                profit = entry_at - exit_price  # Profit from shorting
                profit_pct = (profit / entry_at) * 100

                trades.append(
                    Trade(
                        date=entry_date_str,
                        entry_price=entry_at,
                        exit_price=exit_price,
                        profit=profit,
                        profit_pct=profit_pct,
                        trade_direction=-1,  # SELL
                        holding_period=holding_period,
                        rank=0,
                    )
                )

        # Rank trades by profit (descending)
        trades.sort(key=lambda t: t.profit, reverse=True)
        for i, trade in enumerate(trades):
            trade.rank = i + 1

        return trades

    def identify_top_trades(self, n_trades: int = 10) -> dict[str, list[Trade]]:
        """
        Identify top N most profitable trades for each day.

        Args:
            n_trades: Number of top trades per day (default 10)

        Returns:
            Dictionary: {date_str: [top N trades]}
        """
        if self.historical_data is None:
            self.fetch_historical_data()

        print(f"\nIdentifying top {n_trades} trades per day...")
        df = self._add_technical_features(self.historical_data)

        for day_idx in range(len(df) - 5):  # Leave 5 days for verification
            all_trades = self._simulate_trades(df, day_idx)
            if all_trades:
                top_trades = all_trades[:n_trades]
                date_str = str(df.iloc[day_idx]["date"])
                self.daily_trades[date_str] = top_trades

        print(f"✓ Identified trades for {len(self.daily_trades)} days")
        return self.daily_trades

    def generate_supervised_samples(self) -> list[TradeOpportunity]:
        """
        Generate supervised training samples from top trades.

        Returns:
            List of TradeOpportunity (features -> best_action)
        """
        if not self.daily_trades:
            self.identify_top_trades()

        if self.historical_data is None:
            self.fetch_historical_data()

        print("\nGenerating supervised training samples...")
        df = self._add_technical_features(self.historical_data)

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

        samples = []
        for date_str, trades in self.daily_trades.items():
            # Find matching row in df
            date_obj = pd.to_datetime(date_str)
            mask = df["date"] == date_obj
            if not mask.any():
                continue

            row = df[mask].iloc[0]
            close_price = row["close"]

            # Extract features
            features = {col: float(row[col]) for col in feature_cols if col in row}

            # Determine best action
            buy_trades = [t for t in trades if t.trade_direction == 1]
            sell_trades = [t for t in trades if t.trade_direction == -1]

            best_buy = max(buy_trades, key=lambda t: t.profit_pct) if buy_trades else None
            best_sell = max(sell_trades, key=lambda t: t.profit_pct) if sell_trades else None

            if best_buy and best_sell:
                if best_buy.profit_pct > best_sell.profit_pct:
                    best_action = 1
                    expected_profit = best_buy.profit_pct
                else:
                    best_action = -1
                    expected_profit = best_sell.profit_pct
            elif best_buy:
                best_action = 1
                expected_profit = best_buy.profit_pct
            elif best_sell:
                best_action = -1
                expected_profit = best_sell.profit_pct
            else:
                continue

            # Confidence based on profit consistency
            top_5_profits = sorted([t.profit_pct for t in trades], reverse=True)[:5]
            confidence = np.mean(top_5_profits) / (np.std(top_5_profits) + 1e-6) if len(top_5_profits) > 1 else 0.5

            sample = TradeOpportunity(
                date=date_str,
                close_price=close_price,
                features=features,
                best_action=int(best_action),
                expected_profit=float(expected_profit),
                confidence=float(np.clip(confidence, 0, 1)),
            )
            samples.append(sample)

        self.training_samples = samples
        print(f"✓ Generated {len(samples)} supervised training samples")
        return samples

    def export_supervised_data(self, output_dir: Path | str = "results/supervised_data") -> dict[str, str]:
        """
        Export supervised data to CSV and JSON formats.

        Args:
            output_dir: Directory to save outputs

        Returns:
            Dictionary with paths to exported files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.training_samples:
            self.generate_supervised_samples()

        # Export as CSV
        rows = []
        for sample in self.training_samples:
            row = {"date": sample.date, "close_price": sample.close_price}
            row.update(sample.features)
            row.update(
                {
                    "best_action": sample.best_action,
                    "expected_profit": sample.expected_profit,
                    "confidence": sample.confidence,
                }
            )
            rows.append(row)

        df_supervised = pd.DataFrame(rows)
        csv_path = output_dir / "supervised_training_data.csv"
        df_supervised.to_csv(csv_path, index=False)
        print(f"✓ Exported {len(df_supervised)} samples to {csv_path}")

        # Export trades summary JSON
        trades_summary = {
            "total_days": len(self.daily_trades),
            "total_trades_analyzed": sum(len(t) for t in self.daily_trades.values()),
            "avg_top_trade_profit_pct": float(
                np.mean([max(t, key=lambda x: x.profit_pct).profit_pct for t in self.daily_trades.values()])
            ),
            "max_top_trade_profit_pct": float(
                max([max(t, key=lambda x: x.profit_pct).profit_pct for t in self.daily_trades.values()])
            ),
            "min_top_trade_profit_pct": float(
                min([max(t, key=lambda x: x.profit_pct).profit_pct for t in self.daily_trades.values()])
            ),
        }

        json_path = output_dir / "trade_analysis_summary.json"
        with open(json_path, "w") as f:
            json.dump(trades_summary, f, indent=2)
        print(f"✓ Exported trade summary to {json_path}")

        # Export raw trades for reference
        trades_list = []
        for date_str, trades in self.daily_trades.items():
            for trade in trades[:10]:  # Top 10 only
                trades_list.append(
                    {
                        "date": date_str,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "profit": trade.profit,
                        "profit_pct": trade.profit_pct,
                        "direction": "BUY" if trade.trade_direction == 1 else "SELL",
                        "holding_days": trade.holding_period,
                        "rank": trade.rank,
                    }
                )

        df_trades = pd.DataFrame(trades_list)
        trades_csv_path = output_dir / "top_trades_per_day.csv"
        df_trades.to_csv(trades_csv_path, index=False)
        print(f"✓ Exported {len(df_trades)} top trades to {trades_csv_path}")

        return {
            "supervised_data": str(csv_path),
            "trade_summary": str(json_path),
            "top_trades": str(trades_csv_path),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about analyzed trades."""
        if not self.training_samples:
            self.generate_supervised_samples()

        actions = [s.best_action for s in self.training_samples]
        profits = [s.expected_profit for s in self.training_samples]
        confidences = [s.confidence for s in self.training_samples]

        return {
            "total_samples": len(self.training_samples),
            "buy_signals": sum(1 for a in actions if a == 1),
            "sell_signals": sum(1 for a in actions if a == -1),
            "hold_signals": sum(1 for a in actions if a == 0),
            "avg_expected_profit_pct": float(np.mean(profits)),
            "max_expected_profit_pct": float(np.max(profits)),
            "min_expected_profit_pct": float(np.min(profits)),
            "avg_confidence": float(np.mean(confidences)),
            "data_period_days": self.lookback_days,
            "symbol": self.symbol,
        }


def main():
    """Command-line interface for trade analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze trades and generate supervised training data")
    parser.add_argument("--symbol", default="^NSEI", help="Stock symbol (default: ^NSEI)")
    parser.add_argument("--days", type=int, default=180, help="Historical lookback days (default: 180)")
    parser.add_argument("--top-trades", type=int, default=10, help="Top N trades per day (default: 10)")
    parser.add_argument("--output-dir", default="results/supervised_data", help="Output directory")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")

    args = parser.parse_args()

    analyzer = TradeAnalyzer(symbol=args.symbol, lookback_days=args.days)

    if args.stats:
        analyzer.fetch_historical_data()
        analyzer.identify_top_trades(n_trades=args.top_trades)
        analyzer.generate_supervised_samples()
        stats = analyzer.get_statistics()
        print("\n📊 Trade Analysis Statistics:")
        print(json.dumps(stats, indent=2))
    else:
        analyzer.fetch_historical_data()
        analyzer.identify_top_trades(n_trades=args.top_trades)
        analyzer.generate_supervised_samples()
        paths = analyzer.export_supervised_data(output_dir=args.output_dir)

        print("\n✅ Supervised data generation complete!")
        print("\nExported files:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

        stats = analyzer.get_statistics()
        print("\n📊 Statistics:")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
