from __future__ import annotations

import numpy as np

from .base import BaseModel


class LinearModel(BaseModel):
    """Small online linear regressor trained with stochastic gradient descent."""

    def __init__(
        self,
        n_features: int | None = None,
        learning_rate: float = 0.01,
        l2: float = 0.0,
        fit_intercept: bool = True,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")

        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)
        self.weights: np.ndarray | None = None

        if n_features is not None:
            self._init_weights(n_features)

    def predict(self, x: np.ndarray) -> float:
        features = self._prepare_features(x)
        if self.weights is None:
            self._init_weights(features.size - int(self.fit_intercept))
        return float(np.dot(self.weights, features))

    def update(self, x: np.ndarray, y: float) -> None:
        features = self._prepare_features(x)
        if self.weights is None:
            self._init_weights(features.size - int(self.fit_intercept))

        prediction = float(np.dot(self.weights, features))
        error = float(y) - prediction
        penalty = self.l2 * self.weights
        if self.fit_intercept:
            penalty[-1] = 0.0

        self.weights += self.learning_rate * (error * features - penalty)

    def _init_weights(self, n_features: int) -> None:
        size = int(n_features) + int(self.fit_intercept)
        if size <= 0:
            raise ValueError("n_features must be positive")
        self.weights = np.zeros(size, dtype=float)

    def _prepare_features(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float).reshape(-1)
        if features.size == 0:
            raise ValueError("x must contain at least one feature")
        if self.fit_intercept:
            features = np.concatenate([features, np.ones(1, dtype=float)])
        return features
