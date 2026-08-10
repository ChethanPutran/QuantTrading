"""
Clean Model Predictive Control (MPC) controller implementation.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
import cvxpy as cp

class BaseController(ABC):
    """Decision-making controller interface."""

    @abstractmethod
    def decide(self, state: np.ndarray) -> int:
        """Return chosen action (-1 sell, 0 hold, 1 buy)."""
        raise NotImplementedError()


class MPCConfig(BaseSettings):
    """Configuration for the MPC controller."""

    horizon: int = 5
    transaction_cost: float = 0.0005
    risk_penalty: float = 0.1
    
class MPCController(BaseController):
    """Simple MPC controller that enumerates short action sequences."""

    def __init__(
        self,
        model,
        horizon: int = 5,
        action_space: Optional[List[int]] = None,
        max_position: float = 1.0,
        cost_per_trade: float = 0.001,
        risk_penalty: float = 0.1,
        use_pruning: bool = False,
    ):
        self.model = model
        self.horizon = horizon
        self.action_space = action_space or [-1, 0, 1]
        self.max_position = max_position
        self.cost_per_trade = cost_per_trade
        self.risk_penalty = risk_penalty
        self.use_pruning = use_pruning

        self.current_position = 0.0
        self.total_sequences_evaluated = 0
        self.pruned_sequences = 0

    def decide(self, state: np.ndarray, features: Optional[np.ndarray] = None) -> int:
        sequences = self._generate_sequences()
        self.total_sequences_evaluated += len(sequences)

        best_reward = -np.inf
        best_sequence = None

        for seq in sequences:
            reward = self._simulate_sequence(seq, state, features)
            if reward > best_reward:
                best_reward = reward
                best_sequence = seq

        action = 0 if best_sequence is None else best_sequence[0]

        # update position and clip
        self.current_position += action
        self.current_position = float(np.clip(self.current_position, -self.max_position, self.max_position))
        return int(action)

    def _generate_sequences(self) -> List[List[int]]:
        sequences: List[List[int]] = []

        def rec(cur: List[int]):
            if len(cur) == self.horizon:
                sequences.append(cur.copy())
                return
            for a in self.action_space:
                if cur and abs(a - cur[-1]) > 1:
                    continue
                cur.append(a)
                rec(cur)
                cur.pop()

        rec([])
        return sequences

    def _simulate_sequence(self, sequence: List[int], state: np.ndarray, features: Optional[np.ndarray]) -> float:
        reward = 0.0
        pos = self.current_position
        for a in sequence:
            if a != 0:
                reward -= self.cost_per_trade * abs(a)
            pos += a
            pos = float(np.clip(pos, -self.max_position, self.max_position))
            reward -= self.risk_penalty * (abs(pos) / (self.max_position + 1e-9))
            if hasattr(self.model, 'predict') and features is not None:
                try:
                    pred = self.model.predict(features)
                    reward += float(pred) * pos
                except Exception:
                    pass
        return reward

    def set_position(self, position: float) -> None:
        self.current_position = float(np.clip(position, -self.max_position, self.max_position))

    def reset(self) -> None:
        self.current_position = 0.0
        self.total_sequences_evaluated = 0
        self.pruned_sequences = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            'current_position': self.current_position,
            'total_sequences_evaluated': self.total_sequences_evaluated,
            'pruned_sequences': self.pruned_sequences,
        }

    def forecast(self, expected_returns: np.ndarray, volatility: float, liquidity: float) -> np.ndarray:
        expected_returns = np.asarray(expected_returns, dtype=float).reshape(-1)
        if expected_returns.size == 0:
            expected_returns = np.zeros(self.horizon, dtype=float)
        if expected_returns.size < self.horizon:
            expected_returns = np.pad(expected_returns, (0, self.horizon - expected_returns.size), mode="edge")

        if cp is None:
            return np.clip(expected_returns[: self.horizon] - self.risk_penalty * volatility - self.transaction_cost * liquidity, -1.0, 1.0)

        x = cp.Variable(self.horizon)
        objective = cp.Maximize(cp.sum(x * expected_returns[: self.horizon]) - self.risk_penalty * cp.sum_squares(x) - self.transaction_cost * cp.norm1(x))
        constraints = [x <= 1.0, x >= -1.0]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            solution = x.value if x.value is not None else np.zeros(self.horizon, dtype=float)
            return np.asarray(solution, dtype=float)
        except Exception:
            return np.clip(expected_returns[: self.horizon], -1.0, 1.0)

