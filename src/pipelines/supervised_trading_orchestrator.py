"""
End-to-End Supervised Trading System Training
==============================================
Orchestrates the complete workflow:
1. Analyze historical trades to identify top 10 most profitable per day (6 months)
2. Generate supervised training data from best trades
3. Train supervised models (action classifier + profit regressor)
4. Compare with unsupervised baseline models
5. Generate comprehensive performance report
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.trade_analyzer import TradeAnalyzer
from pipelines.train_supervised_models import SupervisedModelTrainer


@dataclass
class ComparisonReport:
    """Report comparing supervised vs unsupervised models."""

    timestamp: str
    symbol: str
    lookback_days: int
    total_trades_analyzed: int
    top_trades_per_day: int

    # Supervised model metrics
    supervised_samples: int
    supervised_action_accuracy: float
    supervised_profit_rmse: float
    supervised_buy_accuracy: float
    supervised_sell_accuracy: float

    # Unsupervised baseline (if available)
    unsupervised_action_accuracy: float | None = None
    unsupervised_profit_rmse: float | None = None

    # Improvement metrics
    action_accuracy_improvement: float | None = None
    profit_prediction_improvement: float | None = None

    # Recommendation
    recommendation: str = ""


class SupervisedTradingOrchestrator:
    """Orchestrates the complete supervised training workflow."""

    def __init__(
        self,
        symbol: str = "^NSEI",
        lookback_days: int = 180,
        top_trades_per_day: int = 10,
        output_dir: str = "results/supervised_trading",
    ):
        """
        Initialize orchestrator.

        Args:
            symbol: Stock symbol
            lookback_days: Historical window (6 months = 180 days)
            top_trades_per_day: Number of top trades to analyze per day
            output_dir: Base output directory
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.top_trades_per_day = top_trades_per_day
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer: TradeAnalyzer | None = None
        self.trainer: SupervisedModelTrainer | None = None
        self.report: ComparisonReport | None = None

    def step_1_analyze_trades(self) -> dict[str, str]:
        """
        Step 1: Analyze historical trades and identify top performers.

        Returns:
            Dictionary with paths to exported files
        """
        print("\n" + "=" * 70)
        print("STEP 1: ANALYZING HISTORICAL TRADES")
        print("=" * 70)

        self.analyzer = TradeAnalyzer(symbol=self.symbol, lookback_days=self.lookback_days)
        self.analyzer.fetch_historical_data()
        self.analyzer.identify_top_trades(n_trades=self.top_trades_per_day)

        analysis_dir = self.output_dir / "01_trade_analysis"
        analysis_dir.mkdir(exist_ok=True)

        paths = self.analyzer.export_supervised_data(output_dir=analysis_dir)

        stats = self.analyzer.get_statistics()
        print("\n📊 Trade Analysis Summary:")
        print(f"  Total samples analyzed: {stats['total_samples']}")
        print(f"  Buy signals: {stats['buy_signals']}")
        print(f"  Sell signals: {stats['sell_signals']}")
        print(f"  Avg expected profit: {stats['avg_expected_profit_pct']:.2f}%")
        print(f"  Max profit potential: {stats['max_expected_profit_pct']:.2f}%")

        return paths

    def step_2_train_supervised_models(self, supervised_data_csv: str) -> dict[str, str]:
        """
        Step 2: Train supervised models using trade analysis data.

        Args:
            supervised_data_csv: Path to supervised training data

        Returns:
            Dictionary with paths to saved models
        """
        print("\n" + "=" * 70)
        print("STEP 2: TRAINING SUPERVISED MODELS")
        print("=" * 70)

        self.trainer = SupervisedModelTrainer(symbol=self.symbol, lookback_days=self.lookback_days)
        self.trainer.load_supervised_data(supervised_data_csv)
        self.trainer.prepare_data()

        start_time = time.time()
        self.trainer.train_models()
        training_time = time.time() - start_time

        print(f"\n✓ Training completed in {training_time:.2f} seconds")

        models_dir = self.output_dir / "02_supervised_models"
        models_dir.mkdir(exist_ok=True)

        paths = self.trainer.save_models(output_dir=models_dir)
        return paths

    def step_3_generate_report(self, paths: dict[str, dict[str, str]]) -> ComparisonReport:
        """
        Step 3: Generate comprehensive comparison report.

        Args:
            paths: Dictionaries from steps 1 and 2

        Returns:
            Comparison report
        """
        print("\n" + "=" * 70)
        print("STEP 3: GENERATING PERFORMANCE REPORT")
        print("=" * 70)

        # Load trade statistics
        trade_stats = self.analyzer.get_statistics()
        validation_metrics = self.trainer.validate_performance()

        # Extract action classifier accuracy
        action_accuracy = self.trainer.metrics["action_classifier"]["accuracy"]

        # Extract profit regressor metrics
        profit_rmse = self.trainer.metrics["profit_regressor"]["rmse"]

        # Get buy/sell accuracy if available
        buy_acc = validation_metrics.get("BUY_accuracy", None)
        sell_acc = validation_metrics.get("SELL_accuracy", None)

        report = ComparisonReport(
            timestamp=pd.Timestamp.now().isoformat(),
            symbol=self.symbol,
            lookback_days=self.lookback_days,
            total_trades_analyzed=trade_stats["total_rows"] * self.top_trades_per_day,
            top_trades_per_day=self.top_trades_per_day,
            supervised_samples=trade_stats["total_samples"],
            supervised_action_accuracy=float(action_accuracy),
            supervised_profit_rmse=float(profit_rmse),
            supervised_buy_accuracy=float(buy_acc) if buy_acc else 0.0,
            supervised_sell_accuracy=float(sell_acc) if sell_acc else 0.0,
        )

        # Generate recommendations
        recommendations = []

        if action_accuracy > 0.65:
            recommendations.append(
                f"✓ Strong action classifier (accuracy: {action_accuracy:.1%}). Safe to deploy "
                "for buy/sell signal generation."
            )
        elif action_accuracy > 0.55:
            recommendations.append(
                f"○ Moderate action classifier (accuracy: {action_accuracy:.1%}). "
                "Recommend ensemble with other models."
            )
        else:
            recommendations.append(
                f"✗ Weak action classifier (accuracy: {action_accuracy:.1%}). "
                "Model needs more training data or feature engineering."
            )

        if profit_rmse < 2.0:
            recommendations.append(
                f"✓ Excellent profit prediction (RMSE: {profit_rmse:.2f}%). "
                "Can be used for position sizing."
            )
        elif profit_rmse < 5.0:
            recommendations.append(
                f"○ Moderate profit prediction (RMSE: {profit_rmse:.2f}%). "
                "Use cautiously for position sizing."
            )
        else:
            recommendations.append(
                f"✗ High profit prediction error (RMSE: {profit_rmse:.2f}%). "
                "Focus on improving features."
            )

        report.recommendation = " | ".join(recommendations)

        self.report = report
        return report

    def step_4_export_report(self) -> str:
        """
        Step 4: Export comprehensive report to JSON.

        Returns:
            Path to exported report
        """
        if self.report is None:
            raise ValueError("No report generated. Run step_3_generate_report first.")

        report_dir = self.output_dir / "03_reports"
        report_dir.mkdir(exist_ok=True)

        report_path = report_dir / "supervised_training_report.json"

        # Convert dataclass to dict
        report_dict = {
            "timestamp": self.report.timestamp,
            "symbol": self.report.symbol,
            "lookback_days": self.report.lookback_days,
            "total_trades_analyzed": self.report.total_trades_analyzed,
            "top_trades_per_day": self.report.top_trades_per_day,
            "supervised_model_metrics": {
                "samples": self.report.supervised_samples,
                "action_classifier_accuracy": self.report.supervised_action_accuracy,
                "profit_regressor_rmse": self.report.supervised_profit_rmse,
                "buy_signal_accuracy": self.report.supervised_buy_accuracy,
                "sell_signal_accuracy": self.report.supervised_sell_accuracy,
            },
            "recommendations": self.report.recommendation,
        }

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"\n✓ Report exported to {report_path}")
        return str(report_path)

    def run_full_pipeline(self) -> dict[str, Any]:
        """
        Execute the complete pipeline: analyze → train → report.

        Returns:
            Dictionary with all outputs and metrics
        """
        print("\n" + "=" * 70)
        print(f"🎯 SUPERVISED TRADING SYSTEM TRAINING PIPELINE")
        print(f"   Symbol: {self.symbol}")
        print(f"   Lookback: {self.lookback_days} days")
        print(f"   Top trades per day: {self.top_trades_per_day}")
        print("=" * 70)

        start_time = time.time()

        # Step 1: Analyze trades
        analysis_paths = self.step_1_analyze_trades()

        # Step 2: Train supervised models
        model_paths = self.step_2_train_supervised_models(
            supervised_data_csv=analysis_paths["supervised_data"]
        )

        # Step 3: Generate report
        report = self.step_3_generate_report(
            {
                "analysis": analysis_paths,
                "models": model_paths,
            }
        )

        # Step 4: Export report
        report_path = self.step_4_export_report()

        total_time = time.time() - start_time

        # Print final summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\n📊 Final Statistics:")
        print(f"  Action Classifier Accuracy: {report.supervised_action_accuracy:.1%}")
        print(f"  Profit Prediction RMSE: {report.supervised_profit_rmse:.2f}%")
        print(f"  Buy Signal Accuracy: {report.supervised_buy_accuracy:.1%}")
        print(f"  Sell Signal Accuracy: {report.supervised_sell_accuracy:.1%}")

        print(f"\n💡 Recommendations:")
        for rec in report.recommendation.split("|"):
            print(f"  {rec.strip()}")

        print(f"\n⏱️  Total pipeline time: {total_time:.2f} seconds")

        print(f"\n📁 Output directory: {self.output_dir}")

        return {
            "report": report,
            "report_path": report_path,
            "analysis_paths": analysis_paths,
            "model_paths": model_paths,
            "total_time": total_time,
        }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train supervised trading models from historical trade analysis"
    )
    parser.add_argument("--symbol", default="^NSEI", help="Stock symbol (default: ^NSEI)")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Historical lookback days (default: 180 = 6 months)",
    )
    parser.add_argument(
        "--top-trades",
        type=int,
        default=10,
        help="Top N trades per day to analyze (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/supervised_trading",
        help="Output directory",
    )

    args = parser.parse_args()

    orchestrator = SupervisedTradingOrchestrator(
        symbol=args.symbol,
        lookback_days=args.days,
        top_trades_per_day=args.top_trades,
        output_dir=args.output_dir,
    )

    result = orchestrator.run_full_pipeline()

    print("\n" + "=" * 70)
    print("All outputs:")
    for key, path in result["analysis_paths"].items():
        print(f"  Analysis - {key}: {path}")
    for key, path in result["model_paths"].items():
        print(f"  Models - {key}: {path}")
    print(f"  Report: {result['report_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
