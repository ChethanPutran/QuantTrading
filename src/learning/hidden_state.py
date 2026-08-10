from abc import ABC, abstractmethod
import numpy as np


class BaseHiddenStateUpdater(ABC):
    """Updates latent hidden state."""

    @abstractmethod
    def update(self, hidden: np.ndarray, error: float) -> np.ndarray:
        """Updates latent hidden state."""


class HiddenStateUpdater(BaseHiddenStateUpdater):
    """
    Update hidden state based on prediction error.
    
    Purpose:
    - Learns what the model cannot explain
    - Acts as local correction / adaptation
    
    Update rule:
    h_t = λ * h_{t-1} + η * error * f(x)
    
    Where:
    - λ: decay factor (memory)
    - η: learning rate
    - f(x): feature projection
    """
    
    def __init__(
        self,
        hidden_dim: int = 8,
        decay: float = 0.95,
        learning_rate: float = 0.01,
        activation: str = 'tanh'
    ):
        """
        Initialize hidden state updater.
        
        Args:
            hidden_dim: Dimension of hidden state
            decay: Decay factor (memory retention)
            learning_rate: Learning rate for updates
            activation: Activation function ('tanh', 'sigmoid', 'relu')
        """
        self.hidden_dim = hidden_dim
        self.decay = decay
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize projection weights
        self.projection_weights = np.random.randn(hidden_dim) * 0.01
    
    def _activate(self, x: float) -> float:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'relu':
            return max(0, x)
        else:
            return x
    
    def update(
        self,
        hidden: np.ndarray,
        error: float,
        features: np.ndarray = None
    ) -> np.ndarray:
        """
        Update hidden state with error signal.
        
        Args:
            hidden: Current hidden state
            error: Prediction error
            features: Feature vector (for projection)
        
        Returns:
            Updated hidden state
        """
        # Decay existing hidden state
        updated = self.decay * hidden
        
        # Compute correction term
        if features is not None:
            # Project features to hidden dimension
            if len(features) >= self.hidden_dim:
                projection = features[:self.hidden_dim]
            else:
                # Pad with zeros if necessary
                projection = np.pad(features, (0, self.hidden_dim - len(features)))
            
            # Error-weighted update
            correction = self.learning_rate * error * projection
            updated = updated + correction
        else:
            # Simple residual update
            residual = np.ones(self.hidden_dim) * error * self.learning_rate
            updated = updated + residual
        
        # Apply bounds
        updated = np.clip(updated, -10.0, 10.0)
        
        return updated
    
    def get_state_size(self) -> int:
        """Get size of hidden state."""
        return self.hidden_dim
    
    def reset(self) -> np.ndarray:
        """Reset to zero hidden state."""
        return np.zeros(self.hidden_dim)