"""Online Markov regime smoother."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from abc import ABC, abstractmethod



@dataclass
class OnlineHMMRegimeModel:
    n_regimes: int = 3
    smoothing: float = 0.8
    transition_matrix: np.ndarray = field(init=False)
    state_probs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.transition_matrix = np.ones((self.n_regimes, self.n_regimes), dtype=float) / self.n_regimes
        self.state_probs = np.ones(self.n_regimes, dtype=float) / self.n_regimes

    def update(self, observation_probs: np.ndarray) -> np.ndarray:
        observed = np.asarray(observation_probs, dtype=float)
        propagated = self.state_probs @ self.transition_matrix
        blended = self.smoothing * propagated + (1.0 - self.smoothing) * observed
        blended = blended / max(np.sum(blended), 1e-12)
        self.transition_matrix = 0.99 * self.transition_matrix + 0.01 * np.outer(self.state_probs, blended)
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)
        self.state_probs = blended
        return blended
    
class BaseTransitionModel(ABC):
    """Interface for temporal regime models."""

    @abstractmethod
    def update(self, regime_probs: np.ndarray) -> np.ndarray:
        """Returns smoothed regime state"""
        pass


class MarkovTransitionModel(BaseTransitionModel):
    """Smooth regime probabilities with a fixed Markov transition matrix."""

    def __init__(
        self,
        n_regimes: int,
        transition_matrix: np.ndarray | None = None,
        initial_state: np.ndarray | None = None,
        smoothing: float = 0.5,
    ):
        if n_regimes <= 0:
            raise ValueError("n_regimes must be positive")
        if not 0 <= smoothing <= 1:
            raise ValueError("smoothing must be between 0 and 1")

        self.n_regimes = int(n_regimes)
        self.smoothing = float(smoothing)
        self.transition_matrix = self._prepare_transition_matrix(transition_matrix)
        self.state = self._normalise(
            initial_state
            if initial_state is not None
            else np.full(self.n_regimes, 1.0 / self.n_regimes)
        )

    def update(self, regime_probs: np.ndarray) -> np.ndarray:
        observed = self._normalise(regime_probs)
        predicted = self.transition_matrix.T @ self.state
        posterior = predicted * observed
        posterior = self._normalise(posterior)
        self.state = self._normalise(
            self.smoothing * posterior + (1.0 - self.smoothing) * observed
        )
        return self.state.copy()

    def _prepare_transition_matrix(
        self, transition_matrix: np.ndarray | None
    ) -> np.ndarray:
        if transition_matrix is None:
            stay = 0.90
            switch = (1.0 - stay) / max(self.n_regimes - 1, 1)
            matrix = np.full((self.n_regimes, self.n_regimes), switch)
            np.fill_diagonal(matrix, stay)
            if self.n_regimes == 1:
                matrix[0, 0] = 1.0
            return matrix

        matrix = np.asarray(transition_matrix, dtype=float)
        if matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("transition_matrix shape must be (n_regimes, n_regimes)")
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("transition_matrix rows must have positive sums")
        return matrix / row_sums

    def _normalise(self, values: np.ndarray) -> np.ndarray:
        probs = np.asarray(values, dtype=float).reshape(-1)
        if probs.size != self.n_regimes:
            raise ValueError(f"expected {self.n_regimes} regime probabilities")
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(self.n_regimes, 1.0 / self.n_regimes)
        return probs / total
