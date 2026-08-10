"""
Pattern encoder for converting continuous state to discrete memory keys.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BasePatternEncoder(ABC):
    """Encodes state into pattern keys."""

    @abstractmethod
    def encode(self, features: np.ndarray, regime: np.ndarray) -> tuple:
        pass


class PatternEncoder(BasePatternEncoder):
    """
    Encode continuous state into discrete pattern key.
    
    Method:
    - Bin features into discrete levels
    - Append regime information
    - Create hashable tuple
    """
    
    def __init__(
        self,
        feature_bin_size: float = 0.1,
        regime_dim: int = 3
    ):
        """
        Initialize encoder.
        
        Args:
            feature_bin_size: Size of bins for features
            regime_dim: Number of regimes
        """
        self.feature_bin_size = feature_bin_size
        self.regime_dim = regime_dim
    
    def encode(
        self,
        features: np.ndarray,
        regime_probs: np.ndarray
    ) -> Tuple:
        """
        Encode state into pattern key.
        
        Args:
            features: Feature vector of shape (d,)
            regime_probs: Regime probabilities of shape (K,)
        
        Returns:
            Tuple: Hashable pattern key
        """
        # Quantize features
        binned_features = tuple(
            int(np.round(f / self.feature_bin_size))
            for f in features
        )
        
        # Get most likely regime
        regime_id = int(np.argmax(regime_probs))
        
        # Create pattern key
        pattern_key = binned_features + (regime_id,)
        
        return pattern_key
    
    def encode_with_confidence(
        self,
        features: np.ndarray,
        regime_probs: np.ndarray
    ) -> Tuple[Tuple, float]:
        """
        Encode state and return confidence.
        
        Args:
            features: Feature vector
            regime_probs: Regime probabilities
        
        Returns:
            Tuple of (pattern_key, confidence)
        """
        pattern_key = self.encode(features, regime_probs)
        confidence = float(np.max(regime_probs))
        
        return pattern_key, confidence
