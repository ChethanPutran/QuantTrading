"""
Pattern Node - stores experience and statistics for a pattern.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Any, Optional


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


@dataclass(slots=True)
class PatternRecord:
    pattern_id: str
    feature_snapshot: np.ndarray
    hidden_state: np.ndarray
    regime_probs: np.ndarray
    confidence: float = 0.5
    reward_history: list[float] = field(default_factory=list)
    trajectory_history: list[np.ndarray] = field(default_factory=list)
    volatility_context: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    parent_pattern_id: str | None = None

    def score(self, query: np.ndarray) -> float:
        reward_bonus = 0.1 * float(np.mean(self.reward_history)) if self.reward_history else 0.0
        success_rate = self.success_count / max(self.success_count + self.failure_count, 1)
        return _cosine_similarity(self.feature_snapshot, query) + reward_bonus + 0.2 * success_rate + 0.1 * self.confidence


@dataclass
class PatternNode:
    """
    Stores experience for a specific pattern.
    
    Attributes:
        key: Pattern identifier
        hidden_state: Latent state vector for this pattern
        count: Number of times this pattern was seen
        reward_mean: Average reward
        reward_variance: Variance of reward
        hidden_states_history: List of observed hidden states
        children: Child nodes (for branching)
    """
    
    key: tuple
    hidden_state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    count: int = 0
    reward_mean: float = 0.0
    reward_variance: float = 0.0
    error_mean: float = 0.0
    error_variance: float = 0.0
    hidden_states_history: List[np.ndarray] = field(default_factory=list)
    children: Dict[str, 'PatternNode'] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update_idx: int = 0
    
    def update(self, reward: float, error: float, hidden_state: Optional[np.ndarray] = None) -> None:
        """
        Update node with new reward and error.
        
        Args:
            reward: Reward signal (e.g., PnL)
            error: Prediction error
            hidden_state: Updated hidden state
        """
        # Update count
        self.count += 1
        
        # Update reward statistics (Welford's online algorithm)
        n = self.count
        delta = reward - self.reward_mean
        self.reward_mean += delta / n
        delta2 = reward - self.reward_mean
        self.reward_variance += delta * delta2
        
        # Update error statistics
        delta = error - self.error_mean
        self.error_mean += delta / n
        delta2 = error - self.error_mean
        self.error_variance += delta * delta2
        
        # Update hidden state
        if hidden_state is not None:
            alpha = 0.1  # Learning rate
            # Ensure shapes align: pad or truncate incoming hidden_state
            hs = np.asarray(hidden_state, dtype=float).reshape(-1)
            if hs.size != self.hidden_state.size:
                if hs.size < self.hidden_state.size:
                    pad = np.zeros(self.hidden_state.size - hs.size, dtype=float)
                    hs = np.concatenate([hs, pad])
                else:
                    hs = hs[: self.hidden_state.size]

            self.hidden_state = (1 - alpha) * self.hidden_state + alpha * hs
            self.hidden_states_history.append(hs.copy())
            
            # Keep history bounded
            if len(self.hidden_states_history) > 100:
                self.hidden_states_history = self.hidden_states_history[-100:]
    
    def get_reward_variance(self) -> float:
        """Get empirical variance of rewards."""
        if self.count <= 1:
            return 0.0
        return self.reward_variance / max(self.count - 1, 1)
    
    def get_error_variance(self) -> float:
        """Get empirical variance of errors."""
        if self.count <= 1:
            return 0.0
        return self.error_variance / max(self.count - 1, 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            'key': self.key,
            'count': self.count,
            'reward_mean': self.reward_mean,
            'reward_variance': self.get_reward_variance(),
            'error_mean': self.error_mean,
            'error_variance': self.get_error_variance(),
            'hidden_state': self.hidden_state.copy(),
            'n_children': len(self.children)
        }
    
    def create_branch(self, branch_name: str) -> 'PatternNode':
        """
        Create a child branch (for pattern splitting).
        
        Args:
            branch_name: Name of the branch
        
        Returns:
            New PatternNode child
        """
        new_key = self.key + (f"branch_{branch_name}",)
        child = PatternNode(
            key=new_key,
            hidden_state=self.hidden_state.copy(),
            metadata={'parent': self.key}
        )
        self.children[branch_name] = child
        return child
