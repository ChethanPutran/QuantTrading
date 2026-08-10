"""Classification, regression, and trading metrics for online monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Evaluation metric interface."""

    @abstractmethod
    def compute(self, data: dict) -> float:
        pass

@dataclass
class MetricTracker:
    truth: list[int] = field(default_factory=list)
    predictions: list[int] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    pnl: list[float] = field(default_factory=list)

    def update_classification(self, truth: int, prediction: int) -> None:
        self.truth.append(int(truth))
        self.predictions.append(int(prediction))

    def update_trade(self, realized_return: float, cumulative_pnl: float) -> None:
        self.returns.append(float(realized_return))
        self.pnl.append(float(cumulative_pnl))

    def classification_summary(self) -> dict[str, float]:
        if not self.truth:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        truth = np.asarray(self.truth)
        predictions = np.asarray(self.predictions)
        accuracy = float(np.mean(truth == predictions))
        positive_truth = truth > 0
        positive_predictions = predictions > 0
        tp = float(np.sum(positive_truth & positive_predictions))
        fp = float(np.sum(~positive_truth & positive_predictions))
        fn = float(np.sum(positive_truth & ~positive_predictions))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def trading_summary(self) -> dict[str, float]:
        returns = np.asarray(self.returns, dtype=float)
        pnl = np.asarray(self.pnl, dtype=float)
        if returns.size == 0:
            return {"sharpe": 0.0, "win_rate": 0.0, "drawdown": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
        sharpe = float(np.mean(returns) / max(np.std(returns), 1e-12) * np.sqrt(252))
        win_rate = float(np.mean(returns > 0))
        cumulative = np.maximum.accumulate(pnl) if pnl.size else np.array([0.0])
        drawdown = float(np.min((pnl - cumulative) / np.maximum(cumulative, 1e-12))) if pnl.size else 0.0
        profits = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        profit_factor = float(profits / max(losses, 1e-12))
        expectancy = float(np.mean(returns))
        return {"sharpe": sharpe, "win_rate": win_rate, "drawdown": drawdown, "profit_factor": profit_factor, "expectancy": expectancy}
