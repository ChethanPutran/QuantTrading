"""
Gaussian Mixture Model (GMM) for regime detection.
Models market regimes as soft clusters using online EM.

Online Gaussian mixture regime estimator.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


@dataclass
class OnlineGMMRegimeModel:
    n_regimes: int = 3
    feature_dim: int = 14
    learning_rate: float = 0.05
    means: np.ndarray = field(init=False)
    covariances: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.means = np.zeros((self.n_regimes, self.feature_dim), dtype=float)
        self.covariances = np.ones((self.n_regimes, self.feature_dim), dtype=float)
        self.weights = np.ones(self.n_regimes, dtype=float) / self.n_regimes

    def update(self, vector: np.ndarray) -> np.ndarray:
        observation = np.asarray(vector, dtype=float)
        distances = np.linalg.norm(self.means - observation[None, :], axis=1)
        logits = np.exp(-distances)
        probs = logits / max(np.sum(logits), 1e-12)
        for index in range(self.n_regimes):
            rate = self.learning_rate * probs[index]
            self.means[index] = (1.0 - rate) * self.means[index] + rate * observation
            diff = observation - self.means[index]
            self.covariances[index] = (1.0 - rate) * self.covariances[index] + rate * np.maximum(diff * diff, 1e-8)
        self.weights = 0.95 * self.weights + 0.05 * probs
        self.weights = self.weights / np.sum(self.weights)
        return probs



class BaseRegimeModel(ABC):
    """Interface for regime detection models."""

    @abstractmethod
    def update(self, x: np.ndarray) -> np.ndarray:
        """Returns regime probabilities"""
        pass


class GaussianMixtureRegimeModel(BaseRegimeModel):
    """
    Online Gaussian Mixture Model for real-time regime detection.
    Uses online EM for incremental parameter updates.
    
    Model:
    p(x|k) = N(x | μ_k, Σ_k)
    p(x) = Σ_k π_k * N(x | μ_k, Σ_k)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        feature_dim: int = 8,
        learning_rate: float = 0.01,
        covariance_type: str = "diag",
        random_state: int | None = 42,
    ):
        """
        Initialize online GMM.
        
        Args:
            n_regimes: Number of regimes (clusters)
            feature_dim: Dimension of feature vectors
            learning_rate: Learning rate for online EM updates
            covariance_type: 'diag' or 'full'
            random_state: Random seed
        """
        if n_regimes <= 0:
            raise ValueError("n_regimes must be positive")

        self.n_regimes = int(n_regimes)
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize parameters
        self.means = np.random.randn(n_regimes, feature_dim) * 0.1
        self.weights = np.ones(n_regimes) / n_regimes  # π_k (mixing coefficients)
        
        # Initialize covariances
        if covariance_type == 'diag':
            self.covariances = np.ones((n_regimes, feature_dim))
        else:
            self.covariances = np.tile(np.eye(feature_dim), (n_regimes, 1, 1))
        
        self.update_count = 0
        self.responsibilities = np.zeros(n_regimes)

    def _compute_pdf(self, x: np.ndarray, cluster: int) -> float:
        """
        Compute Gaussian PDF for cluster k.
        
        Args:
            x: Feature vector
            cluster: Cluster index
        
        Returns:
            Probability density
        """
        mean = self.means[cluster]
        
        if self.covariance_type == 'diag':
            cov = self.covariances[cluster]
            diff = x - mean
            exponent = -0.5 * np.sum((diff ** 2) / (cov + 1e-6))
            det = np.prod(cov + 1e-6)
            norm = 1.0 / np.sqrt((2 * np.pi) ** self.feature_dim * det + 1e-10)
        else:
            cov = self.covariances[cluster]
            diff = x - mean
            try:
                inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(self.feature_dim))
                exponent = -0.5 * diff @ inv_cov @ diff
                det = np.linalg.det(cov + 1e-6 * np.eye(self.feature_dim))
                norm = 1.0 / np.sqrt((2 * np.pi) ** self.feature_dim * (det + 1e-10))
            except:
                return 1e-6
        
        return norm * np.exp(exponent)

    def update(self, x: np.ndarray) -> np.ndarray:
        """
        Online EM update with new data point.
        
        Args:
            x: Feature vector of shape (feature_dim,)
        
        Returns:
            Responsibilities (posterior probabilities) of shape (n_regimes,)
        """
        obs = np.asarray(x, dtype=float)
        if obs.size == 0:
            raise ValueError("x must contain at least one feature")
        
        # E-step: compute responsibilities
        likelihoods = np.array([
            self._compute_pdf(obs, k) for k in range(self.n_regimes)
        ])
        
        # Weight by mixing coefficients
        weighted_likelihoods = self.weights * likelihoods
        
        # Normalize to get responsibilities (posterior probabilities)
        evidence = np.sum(weighted_likelihoods) + 1e-10
        self.responsibilities = weighted_likelihoods / evidence
        
        # M-step: update parameters (online with decaying learning rate)
        alpha = self.learning_rate / (1.0 + 0.001 * self.update_count)
        
        # Update weights (mixing coefficients)
        self.weights = (1 - alpha) * self.weights + alpha * self.responsibilities
        self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        # Update means
        for k in range(self.n_regimes):
            self.means[k] = ((1 - alpha * self.responsibilities[k]) * self.means[k] +
                            alpha * self.responsibilities[k] * obs)
        
        # Update covariances
        for k in range(self.n_regimes):
            diff = obs - self.means[k]
            
            if self.covariance_type == 'diag':
                new_cov = diff ** 2
                self.covariances[k] = ((1 - alpha * self.responsibilities[k]) * self.covariances[k] +
                                       alpha * self.responsibilities[k] * new_cov)
                # Floor to avoid collapse
                self.covariances[k] = np.maximum(self.covariances[k], 1e-6)
            else:
                new_cov = np.outer(diff, diff)
                self.covariances[k] = ((1 - alpha * self.responsibilities[k]) * self.covariances[k] +
                                      alpha * self.responsibilities[k] * new_cov)
        
        self.update_count += 1
        
        return self.responsibilities.copy()
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Predict regime for feature vector.
        
        Args:
            x: Feature vector
        
        Returns:
            Tuple of (responsibilities, regime_id)
        """
        probs = self.update(x)
        regime_id = np.argmax(probs)
        return probs, regime_id
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'means': self.means.copy(),
            'weights': self.weights.copy(),
            'covariances': self.covariances.copy() if isinstance(self.covariances, np.ndarray) else self.covariances,
            'update_count': self.update_count
        }
    
    def set_params(self, params: dict) -> None:
        """Set model parameters."""
        self.means = params['means'].copy()
        self.weights = params['weights'].copy()
        self.covariances = params['covariances'].copy() if isinstance(params['covariances'], np.ndarray) else params['covariances']
        self.update_count = params.get('update_count', 0)
    
    def reset(self) -> None:
        """Reset to random initialization."""
        self.means = np.random.randn(self.n_regimes, self.feature_dim) * 0.1
        self.weights = np.ones(self.n_regimes) / self.n_regimes
        
        if self.covariance_type == 'diag':
            self.covariances = np.ones((self.n_regimes, self.feature_dim))
        else:
            self.covariances = np.tile(np.eye(self.feature_dim), (self.n_regimes, 1, 1))
        
        self.update_count = 0
        self.responsibilities = np.zeros(self.n_regimes)
