"""Online learner that updates models, memory, and hidden-state statistics after each trade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....src.trading_system.core.events import FeedbackEvent, TradeAction
from ....src.trading_system.memory.patterns import PatternDB
from ....src.trading_system.models.base import ModelBundle


@dataclass
class OnlineLearner:
    model_bundle: ModelBundle
    pattern_db: PatternDB

    def update(self, feedback: FeedbackEvent) -> None:
        feature_row = feedback.state.features
        action_label = {TradeAction.BUY: 1, TradeAction.SELL: -1, TradeAction.HOLD: 0}[feedback.action]
        self.model_bundle.update(feature_row, action_label, feedback.reward)
        pattern = self.pattern_db.get_or_create(feature_row, feedback.state.hidden_state, feedback.state.regime_probs)
        success = feedback.reward >= 0.0
        self.pattern_db.update_feedback(pattern.pattern_id, feedback.reward, feedback.state.features, success)
        pattern.hidden_state = 0.9 * pattern.hidden_state + 0.1 * feedback.state.hidden_state
        pattern.regime_probs = 0.9 * pattern.regime_probs + 0.1 * feedback.state.regime_probs
        pattern.confidence = float(np.clip(pattern.confidence + (0.02 if success else -0.02), 0.05, 0.99))
