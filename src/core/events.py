"""
Event system for async architecture.
Defines all event types used in the event-driven architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence
import numpy as np
from enum import Enum

from memory.pattern import PatternRecord
from models.base import ModelOutput


class EventType(str, Enum):
    TICK = "TICK"
    FEATURE = "FEATURE"
    REGIME = "REGIME"
    PATTERN = "PATTERN"
    PREDICTION = "PREDICTION"
    ACTION = "ACTION"
    DECISION = "DECISION"
    EXECUTION = "EXECUTION"
    FEEDBACK = "FEEDBACK"
    LEARNING = "LEARNING"

class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(slots=True)
class MarketEvent:
    timestamp: float
    symbol: str
    metadata: Dict[Any, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TickEvent(MarketEvent):
    price: float = 0.0
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    spread: float | None = None
    imbalance: float | None = None


@dataclass(slots=True)
class StateVector:
    timestamp: float
    symbol: str
    timeframe: str
    features: np.ndarray
    filtered_price: float
    volatility: float
    regime_probs: np.ndarray
    hidden_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.features, dtype=float)

@dataclass(slots=True)
class FeatureEvent:
    state: StateVector


@dataclass(slots=True)
class RegimeEvent:
    """Regime probabilities from GMM + HMM."""
    gmm_probs: np.ndarray  # Shape: (K,) - probability per cluster
    hmm_probs: np.ndarray  # Shape: (K,) - transition-smoothed probs
    regime_id: int  # Most likely regime
    timestamp: float
    state: StateVector
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatternEvent:
    """Pattern retrieval from memory."""
    state: StateVector
    pattern_key: tuple
    hidden_state: np.ndarray
    pattern: PatternRecord
    node_stats: Dict[str, Any]  # count, avg_reward, variance, etc.
    timestamp: float


@dataclass(slots=True)
class PredictionEvent:
    """Prediction from model."""
    state: StateVector
    model_output: ModelOutput
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass(slots=True)
class ActionEvent:
    state: StateVector
    action: TradeAction 
    confidence: float  # How confident is MPC?
    expected_reward: float
    expected_risk: float
    planned_trajectory: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))


@dataclass(slots=True)
class DecisionEvent(ActionEvent):
    decision_weights: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionEvent:
    """Result of trade execution."""
    state: StateVector
    action: TradeAction
    quantity: float
    price: float
    transaction_cost: float
    position_after: float
    cash_after: float


@dataclass(slots=True)
class FeedbackEvent:
    """Feedback on executed trade."""
    state: StateVector
    reward: float  # PnL or other reward signal
    action: TradeAction
    actual_return: float  # Realized return
    realized_pnl: float
    prediction_error: float # |actual_return - predicted_return|
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEvent:
    """Event triggered by learning module."""
    model_update_magnitude: float  # |Δw|
    hidden_state_update: float  # |Δh|
    pattern_branching_triggered: bool
    timestamp: float


def ensure_ndarray(values: Sequence[float] | np.ndarray, *, dtype: type[float] = float) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


class EventHandler:
    """Base class for event handlers."""
    async def handle(self, event: Any) -> None:
        raise NotImplementedError("EventHandler subclasses must implement the handle method.")