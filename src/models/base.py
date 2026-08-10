
"""Prediction model bundle with optional sklearn/torch backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(slots=True)
class ModelOutput:
    action_logits: dict[str, float]
    predicted_return: float
    predicted_std: float

class BaseModel(ABC):
    """Abstract base for all predictive models."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def update(self, x: np.ndarray, y: float) -> None:
        pass


class _TorchSequenceModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        if nn is None:
            raise RuntimeError("torch is not available")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs):  # type: ignore[override]
        return self.network(inputs)

@dataclass
class ModelBundle:
    feature_dim: int
    action_model: Any = field(default_factory=DecisionTreeClassifier)
    boost_action_model: Any = field(default_factory=GradientBoostingClassifier)
    svm_model: Any = field(default_factory=lambda: SVC(probability=True))
    regression_tree: Any = field(default_factory=DecisionTreeRegressor)
    boosting_regressor: Any = field(default_factory=GradientBoostingRegressor)
    online_action_model: Any = field(default_factory=lambda: SGDClassifier(loss="log_loss", random_state=42))
    online_regressor: Any = field(default_factory=lambda: SGDRegressor(random_state=42))
    torch_model: Any | None = None
    fitted: bool = False
    online_ready: bool = False

    def __post_init__(self) -> None:
        if self.torch_model is None and torch is not None:
            self.torch_model = _TorchSequenceModel(self.feature_dim)

    def fit(self, features: np.ndarray, action_labels: np.ndarray, returns: np.ndarray) -> None:
        features = np.asarray(features, dtype=float)
        action_labels = np.asarray(action_labels, dtype=int)
        returns = np.asarray(returns, dtype=float)
        self.action_model.fit(features, action_labels)
        self.boost_action_model.fit(features, action_labels)
        self.svm_model.fit(features, action_labels)
        self.regression_tree.fit(features, returns)
        self.boosting_regressor.fit(features, returns)
        self.online_action_model.partial_fit(features, action_labels, classes=np.array([-1, 0, 1]))
        self.online_regressor.partial_fit(features, returns)
        self.fitted = True
        self.online_ready = True

    def update(self, feature_row: np.ndarray, action_label: int, realized_return: float) -> None:
        feature_row = np.asarray(feature_row, dtype=float).reshape(1, -1)
        label = np.asarray([action_label], dtype=int)
        self.online_action_model.partial_fit(feature_row, label, classes=np.array([-1, 0, 1]))
        self.online_regressor.partial_fit(feature_row, np.asarray([realized_return], dtype=float))
        self.online_ready = True
        if hasattr(self.regression_tree, "fit"):
            self.regression_tree.fit(feature_row, np.asarray([realized_return], dtype=float))

    def predict(self, feature_row: np.ndarray) -> ModelOutput:
        feature_row = np.asarray(feature_row, dtype=float).reshape(1, -1)
        if not self.fitted and not self.online_ready:
            return ModelOutput(
                action_logits={"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34},
                predicted_return=0.0,
                uncertainty=1.0,
            )
        action_votes: dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        if self.fitted:
            for model in (self.action_model, self.boost_action_model, self.svm_model):
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(feature_row)[0]
                    classes = getattr(model, "classes_", np.array([-1, 0, 1]))
                    for class_id, probability in zip(classes, probabilities):
                        label = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(class_id)]
                        action_votes[label] += float(probability)
        if self.online_ready:
            online_probabilities = self.online_action_model.predict_proba(feature_row)[0]
            for class_id, probability in zip(self.online_action_model.classes_, online_probabilities):
                label = {1: "BUY", -1: "SELL", 0: "HOLD"}[int(class_id)]
                action_votes[label] += float(probability)
        predicted_return = 0.0
        if self.fitted:
            predicted_return = float(self.boosting_regressor.predict(feature_row)[0])
        if self.online_ready:
            online_prediction = float(self.online_regressor.predict(feature_row)[0])
            predicted_return = online_prediction if not self.fitted else 0.5 * predicted_return + 0.5 * online_prediction
        uncertainty = float(np.std(list(action_votes.values())))
        return ModelOutput(action_logits=action_votes, predicted_return=predicted_return, uncertainty=uncertainty)
