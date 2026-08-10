"""
Supervised Model Retraining Pipeline
====================================
Retrains trading models using supervised data from best historical trades.

This pipeline:
1. Loads supervised training data from trade analysis
2. Scales features consistently
3. Trains models with supervised labels (best_action, expected_profit)
4. Performs validation on held-out data
5. Compares performance: unsupervised vs supervised models
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class SupervisedTrainingResult:
    """Results from supervised model training."""
    symbol: str
    total_samples: int
    train_samples: int
    val_samples: int
    feature_count: int
    used_lstm: bool
    used_xgboost: bool
    action_classifier_accuracy: float
    action_classifier_f1: float
    profit_regressor_rmse: float
    profit_regressor_r2: float
    validation_buy_accuracy: float
    validation_sell_accuracy: float
    avg_expected_profit: float
    training_time: float


class SupervisedModelTrainer:
    """Trains models using supervised trade data."""

    def __init__(
        self,
        symbol: str = "^NSEI",
        lookback_days: int = 180,
        test_size: float = 0.2,
    ):
        """
        Initialize supervised trainer.

        Args:
            symbol: Stock symbol
            lookback_days: Historical window size
            test_size: Train/validation split ratio
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.test_size = test_size

        self.supervised_data: pd.DataFrame | None = None
        self.X_train: np.ndarray | None = None
        self.y_action_train: np.ndarray | None = None
        self.y_profit_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_action_val: np.ndarray | None = None
        self.y_profit_val: np.ndarray | None = None

        self.feature_scaler: StandardScaler | None = None
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}

    def load_supervised_data(self, csv_path: Path | str) -> pd.DataFrame:
        """
        Load supervised training data from CSV.

        Args:
            csv_path: Path to supervised_training_data.csv

        Returns:
            DataFrame with features and labels
        """
        print(f"Loading supervised data from {csv_path}...")
        self.supervised_data = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.supervised_data)} samples")
        return self.supervised_data

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.

        Returns:
            Tuple of (X, y_action, y_profit)
        """
        if self.supervised_data is None:
            raise ValueError("No supervised data loaded. Call load_supervised_data first.")

        df = self.supervised_data.copy()

        # Identify feature columns (everything except special columns)
        skip_cols = {
            "date",
            "close_price",
            "best_action",
            "expected_profit",
            "confidence",
        }
        feature_cols = [col for col in df.columns if col not in skip_cols]

        print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

        # Extract arrays
        X = df[feature_cols].fillna(0).values
        y_action = df["best_action"].values
        y_profit = df["expected_profit"].values

        # Split data
        X_train, X_val, y_act_train, y_act_val, y_prof_train, y_prof_val = train_test_split(
            X, y_action, y_profit, test_size=self.test_size, random_state=42
        )

        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.y_action_train = y_act_train
        self.y_action_val = y_act_val
        self.y_profit_train = y_prof_train
        self.y_profit_val = y_prof_val

        print(f"✓ Data prepared: {len(X_train)} train, {len(X_val)} val")
        # Map actions (-1, 0, 1) to (0, 1, 2) for bincount
        actions_mapped = y_act_train + 1  # Convert -1->0, 0->1, 1->2
        action_counts = np.bincount(actions_mapped.astype(int))
        action_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
        dist_str = " | ".join([f"{action_labels[i]}:{action_counts[i] if i < len(action_counts) else 0}" for i in range(3)])
        print(f"  Action distribution (train): {dist_str}")
        print(f"  Profit range: [{y_prof_train.min():.2f}, {y_prof_train.max():.2f}]%")

        return X_train_scaled, y_act_train, y_prof_train

    def train_models(self) -> dict[str, Any]:
        """
        Train supervised models for action classification and profit prediction.

        Returns:
            Dictionary of trained models
        """
        if self.X_train is None:
            self.prepare_data()

        print("\nTraining supervised models...")

        # ==================== Action Classification ====================
        # Predict if action should be BUY (+1), SELL (-1), or HOLD (0)

        print("\n1. Training Action Classifier...")

        # Logistic Regression for action
        lr_action = LogisticRegression(max_iter=1000, random_state=42)
        lr_action.fit(self.X_train, self.y_action_train)
        y_pred_action = lr_action.predict(self.X_val)
        action_acc = accuracy_score(self.y_action_val, y_pred_action)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_action_val, y_pred_action, average="weighted", zero_division=0
        )

        print(f"  Logistic Regression Accuracy: {action_acc:.4f}, F1: {f1:.4f}")

        # Decision Tree for action
        dt_action = DecisionTreeClassifier(max_depth=8, random_state=42)
        dt_action.fit(self.X_train, self.y_action_train)
        y_pred_dt_action = dt_action.predict(self.X_val)
        dt_action_acc = accuracy_score(self.y_action_val, y_pred_dt_action)
        print(f"  Decision Tree Accuracy: {dt_action_acc:.4f}")

        # Gradient Boosting for action
        gb_action = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb_action.fit(self.X_train, self.y_action_train)
        y_pred_gb_action = gb_action.predict(self.X_val)
        gb_action_acc = accuracy_score(self.y_action_val, y_pred_gb_action)
        print(f"  Gradient Boosting Accuracy: {gb_action_acc:.4f}")

        # Choose best action model
        best_action_model = max(
            [("logistic_regression", lr_action, action_acc), ("decision_tree", dt_action, dt_action_acc),
             ("gradient_boosting", gb_action, gb_action_acc)],
            key=lambda x: x[2],
        )
        print(f"  ✓ Best action model: {best_action_model[0]} (acc={best_action_model[2]:.4f})")

        # ==================== Profit Prediction ====================
        # Predict expected profit percentage

        print("\n2. Training Profit Regressor...")

        # Linear Regression for profit
        lr_profit = LinearRegression()
        lr_profit.fit(self.X_train, self.y_profit_train)
        y_pred_profit = lr_profit.predict(self.X_val)
        lr_rmse = np.sqrt(mean_squared_error(self.y_profit_val, y_pred_profit))
        lr_r2 = 1 - np.sum((self.y_profit_val - y_pred_profit) ** 2) / np.sum((self.y_profit_val - np.mean(self.y_profit_val)) ** 2)
        print(f"  Linear Regression RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")

        # Decision Tree for profit
        dt_profit = DecisionTreeRegressor(max_depth=8, random_state=42)
        dt_profit.fit(self.X_train, self.y_profit_train)
        y_pred_dt_profit = dt_profit.predict(self.X_val)
        dt_rmse = np.sqrt(mean_squared_error(self.y_profit_val, y_pred_dt_profit))
        dt_r2 = 1 - np.sum((self.y_profit_val - y_pred_dt_profit) ** 2) / np.sum((self.y_profit_val - np.mean(self.y_profit_val)) ** 2)
        print(f"  Decision Tree RMSE: {dt_rmse:.4f}, R²: {dt_r2:.4f}")

        # Gradient Boosting for profit
        gb_profit = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_profit.fit(self.X_train, self.y_profit_train)
        y_pred_gb_profit = gb_profit.predict(self.X_val)
        gb_rmse = np.sqrt(mean_squared_error(self.y_profit_val, y_pred_gb_profit))
        gb_r2 = 1 - np.sum((self.y_profit_val - y_pred_gb_profit) ** 2) / np.sum((self.y_profit_val - np.mean(self.y_profit_val)) ** 2)
        print(f"  Gradient Boosting RMSE: {gb_rmse:.4f}, R²: {gb_r2:.4f}")

        # Choose best profit model
        best_profit_model = min(
            [("linear_regression", lr_profit, lr_rmse), ("decision_tree", dt_profit, dt_rmse),
             ("gradient_boosting", gb_profit, gb_rmse)],
            key=lambda x: x[2],
        )
        print(f"  ✓ Best profit model: {best_profit_model[0]} (rmse={best_profit_model[2]:.4f})")

        self.models = {
            "action_classifier": best_action_model[1],
            "action_classifier_name": best_action_model[0],
            "profit_regressor": best_profit_model[1],
            "profit_regressor_name": best_profit_model[0],
            "feature_scaler": self.feature_scaler,
        }

        # Store metrics
        self.metrics = {
            "action_classifier": {
                "accuracy": float(best_action_model[2]),
                "model_type": best_action_model[0],
            },
            "profit_regressor": {
                "rmse": float(best_profit_model[2]),
                "r2": float(best_profit_model[3] if len(best_profit_model) > 3 else (
                    gb_r2 if best_profit_model[0] == "gradient_boosting" else (
                        dt_r2 if best_profit_model[0] == "decision_tree" else lr_r2
                    )
                )),
                "model_type": best_profit_model[0],
            },
        }

        return self.models

    def validate_performance(self) -> dict[str, Any]:
        """
        Validate model performance on specific action types.

        Returns:
            Validation metrics
        """
        if not self.models:
            self.train_models()

        classifier = self.models["action_classifier"]
        y_pred = classifier.predict(self.X_val)

        metrics = {}
        for action in [-1, 0, 1]:
            action_name = "BUY" if action == 1 else ("SELL" if action == -1 else "HOLD")
            mask = self.y_action_val == action
            if mask.sum() > 0:
                acc = accuracy_score(self.y_action_val[mask], y_pred[mask])
                metrics[f"{action_name}_accuracy"] = float(acc)
                metrics[f"{action_name}_samples"] = int(mask.sum())

        return metrics

    def save_models(self, output_dir: Path | str = "results/supervised_models") -> dict[str, str]:
        """
        Save trained models to disk.

        Args:
            output_dir: Directory to save models

        Returns:
            Dictionary with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.models:
            self.train_models()

        # Save ensemble model
        models_path = output_dir / "supervised_models.joblib"
        joblib.dump(self.models, models_path)
        print(f"✓ Saved models to {models_path}")

        # Save metrics
        metrics_path = output_dir / "training_metrics.json"
        metrics_to_save = {
            "symbol": self.symbol,
            "lookback_days": self.lookback_days,
            "total_samples": len(self.supervised_data) if self.supervised_data is not None else 0,
            "train_samples": int(len(self.X_train) if self.X_train is not None else 0),
            "val_samples": int(len(self.X_val) if self.X_val is not None else 0),
            "models": self.metrics,
            "validation": self.validate_performance(),
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"✓ Saved metrics to {metrics_path}")

        return {
            "models": str(models_path),
            "metrics": str(metrics_path),
        }


def main():
    """Command-line interface for supervised model training."""
    parser = argparse.ArgumentParser(description="Train models using supervised trade data")
    parser.add_argument("--symbol", default="^NSEI", help="Stock symbol")
    parser.add_argument("--supervised-data", required=True, help="Path to supervised_training_data.csv")
    parser.add_argument("--output-dir", default="results/supervised_models", help="Output directory")

    args = parser.parse_args()

    trainer = SupervisedModelTrainer(symbol=args.symbol)
    trainer.load_supervised_data(args.supervised_data)
    trainer.prepare_data()
    trainer.train_models()

    print("\n" + "=" * 60)
    print("VALIDATION PERFORMANCE")
    print("=" * 60)
    validation_metrics = trainer.validate_performance()
    for key, val in validation_metrics.items():
        if "accuracy" in key:
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")

    paths = trainer.save_models(output_dir=args.output_dir)

    print("\n✅ Supervised model training complete!")
    print("\nExported files:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
