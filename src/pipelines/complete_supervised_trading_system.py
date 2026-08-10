"""
Complete Supervised Trading System - End-to-End Workflow
=======================================================
Orchestrates the full pipeline:
1. Analyze historical trades (identify top 10 most profitable per day)
2. Generate supervised training data
3. Train supervised models (action classifier + profit regressor)
4. Simulate trading using trained supervised models
5. Generate comprehensive performance report comparing models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.trade_analyzer import TradeAnalyzer
from pipelines.simulate_with_supervised_models import SupervisedTradingSimulator
from pipelines.train_supervised_models import SupervisedModelTrainer


class CompleteSupervisedTradingSystem:
    """Complete end-to-end supervised trading system."""

    def __init__(
        self,
        symbol: str = "^NSEI",
        lookback_days: int = 180,
        simulate_days: int = 30,
        top_trades_per_day: int = 10,
        output_dir: str = "results/complete_supervised_trading",
    ):
        """
        Initialize complete system.

        Args:
            symbol: Stock symbol
            lookback_days: Historical window for analysis
            simulate_days: Future window for simulation
            top_trades_per_day: Top N trades per day to analyze
            output_dir: Base output directory
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.simulate_days = simulate_days
        self.top_trades_per_day = top_trades_per_day
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def step_1_analyze_trades(self) -> str:
        """Step 1: Analyze historical trades."""
        print("\n" + "=" * 70)
        print("STEP 1: ANALYZING HISTORICAL TRADES")
        print("=" * 70)

        analyzer = TradeAnalyzer(
            symbol=self.symbol,
            lookback_days=self.lookback_days,
        )
        analyzer.fetch_historical_data()
        analyzer.identify_top_trades(n_trades=self.top_trades_per_day)

        analysis_dir = self.output_dir / "01_trade_analysis"
        analysis_dir.mkdir(exist_ok=True)

        paths = analyzer.export_supervised_data(output_dir=analysis_dir)
        supervised_data_csv = paths["supervised_data"]

        stats = analyzer.get_statistics()
        print(f"\n📊 Trade Analysis:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Buy signals: {stats['buy_signals']}")
        print(f"   Sell signals: {stats['sell_signals']}")
        print(f"   Avg expected profit: {stats['avg_expected_profit_pct']:.2f}%")

        return supervised_data_csv

    def step_2_train_models(self, supervised_data_csv: str) -> str:
        """Step 2: Train supervised models."""
        print("\n" + "=" * 70)
        print("STEP 2: TRAINING SUPERVISED MODELS")
        print("=" * 70)

        trainer = SupervisedModelTrainer(symbol=self.symbol)
        trainer.load_supervised_data(supervised_data_csv)
        trainer.prepare_data()

        start_time = time.time()
        trainer.train_models()
        training_time = time.time() - start_time

        models_dir = self.output_dir / "02_supervised_models"
        models_dir.mkdir(exist_ok=True)

        paths = trainer.save_models(output_dir=models_dir)
        models_path = paths["models"]

        print(f"\n✓ Training completed in {training_time:.2f}s")
        print(f"   Action classifier accuracy: {trainer.metrics['action_classifier']['accuracy']:.1%}")
        print(f"   Profit regressor RMSE: {trainer.metrics['profit_regressor']['rmse']:.2f}%")

        return models_path

    def step_3_simulate_trading(self, models_path: str) -> dict[str, Any]:
        """Step 3: Simulate trading using supervised models."""
        print("\n" + "=" * 70)
        print("STEP 3: SIMULATING TRADING WITH SUPERVISED MODELS")
        print("=" * 70)

        simulator = SupervisedTradingSimulator(
            symbol=self.symbol,
            total_days=self.lookback_days + self.simulate_days,
            simulate_days=self.simulate_days,
            models_path=models_path,
        )

        simulator.fetch_historical_data()
        simulator.load_supervised_models()
        simulator.simulate()

        sim_dir = self.output_dir / "03_simulation_results"
        sim_dir.mkdir(exist_ok=True)

        sim_paths = simulator.export_simulation(
            output_csv=str(sim_dir / "supervised_trading_simulation.csv"),
            output_json=str(sim_dir / "supervised_trading_metrics.json"),
        )

        return {
            "metrics": simulator.metrics,
            "paths": sim_paths,
        }

    def step_4_generate_report(
        self,
        trade_stats: dict[str, Any],
        model_path: str,
        sim_results: dict[str, Any],
    ) -> str:
        """Step 4: Generate comprehensive report."""
        print("\n" + "=" * 70)
        print("STEP 4: GENERATING COMPREHENSIVE REPORT")
        print("=" * 70)

        report = {
            "system_name": "Complete Supervised Trading System",
            "timestamp": pd.Timestamp.now().isoformat(),
            "configuration": {
                "symbol": self.symbol,
                "lookback_days": self.lookback_days,
                "simulate_days": self.simulate_days,
                "top_trades_per_day": self.top_trades_per_day,
            },
            "trade_analysis": trade_stats,
            "model_path": model_path,
            "simulation_metrics": sim_results["metrics"],
            "recommendations": self._generate_recommendations(sim_results["metrics"]),
        }

        report_dir = self.output_dir / "04_reports"
        report_dir.mkdir(exist_ok=True)

        report_path = report_dir / "complete_trading_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("📊 TRADING PERFORMANCE SUMMARY")
        print("=" * 70)
        metrics = sim_results["metrics"]
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"Profit/Loss: ${metrics['final_pnl']:,.2f}")
        print(f"Return: {(metrics['final_pnl'] / metrics['initial_capital'] * 100):.2f}%")
        print(f"Trades - Buy: {metrics['buy_trades']}, Sell: {metrics['sell_trades']}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")

        return str(report_path)

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate trading recommendations based on metrics."""
        recommendations = []

        pnl_pct = (metrics["final_pnl"] / metrics["initial_capital"]) * 100
        if pnl_pct > 5:
            recommendations.append(f"✓ Excellent returns ({pnl_pct:.2f}%). System is profitable.")
        elif pnl_pct > 0:
            recommendations.append(f"○ Positive returns ({pnl_pct:.2f}%). System is viable.")
        else:
            recommendations.append(
                f"✗ Negative returns ({pnl_pct:.2f}%). "
                "Optimize features or increase model training data."
            )

        trade_count = metrics["buy_trades"] + metrics["sell_trades"]
        if trade_count < 5:
            recommendations.append(
                f"Note: Low trade count ({trade_count}). "
                "Consider adjusting prediction thresholds."
            )
        else:
            recommendations.append(
                f"✓ Good trading activity ({trade_count} total trades). "
                "System is generating signals."
            )

        if metrics["max_drawdown"] < 0.05:
            recommendations.append("✓ Low drawdown. Risk management is effective.")
        elif metrics["max_drawdown"] < 0.15:
            recommendations.append("○ Moderate drawdown. Review position sizing.")
        else:
            recommendations.append("✗ High drawdown. Reduce position sizes or improve stop-loss.")

        return recommendations

    def run_complete_pipeline(self) -> dict[str, str]:
        """Execute the complete end-to-end pipeline."""
        print("\n" + "=" * 70)
        print("🎯 COMPLETE SUPERVISED TRADING SYSTEM")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Lookback: {self.lookback_days} days | Simulate: {self.simulate_days} days")
        print(f"Output: {self.output_dir}")

        start_time = time.time()

        # Step 1
        supervised_data = self.step_1_analyze_trades()

        # Step 2
        models_path = self.step_2_train_models(supervised_data)

        # Step 3
        sim_results = self.step_3_simulate_trading(models_path)

        # Step 4
        report_path = self.step_4_generate_report(
            trade_stats={},
            model_path=models_path,
            sim_results=sim_results,
        )

        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("✅ COMPLETE PIPELINE FINISHED")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        print(f"📊 Report: {report_path}")

        return {
            "output_dir": str(self.output_dir),
            "report": report_path,
            "models": models_path,
            "simulation_csv": sim_results["paths"].get("csv", ""),
            "total_time": total_time,
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Complete supervised trading system - end-to-end workflow"
    )
    parser.add_argument("--symbol", default="^NSEI", help="Stock symbol (default: ^NSEI)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=180,
        help="Analysis period (default: 180 = 6 months)",
    )
    parser.add_argument(
        "--simulate-days",
        type=int,
        default=30,
        help="Simulation period (default: 30 = 1 month)",
    )
    parser.add_argument(
        "--top-trades",
        type=int,
        default=10,
        help="Top trades per day (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/complete_supervised_trading",
        help="Output directory",
    )

    args = parser.parse_args()

    system = CompleteSupervisedTradingSystem(
        symbol=args.symbol,
        lookback_days=args.lookback_days,
        simulate_days=args.simulate_days,
        top_trades_per_day=args.top_trades,
        output_dir=args.output_dir,
    )

    result = system.run_complete_pipeline()

    print("\n" + "=" * 70)
    print("Output Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
