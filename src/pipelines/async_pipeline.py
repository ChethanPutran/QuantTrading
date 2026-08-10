"""
Async pipeline orchestration for event-driven trading system.
Coordinates all async modules through the event bus.
"""

import asyncio
from datetime import datetime
import logging
from typing import Callable, Dict, Any, Optional
from anyio import Path
import numpy as np
import pandas as pd

from config.settings import PipelineSettings
from control.mpc import MPCController
from core.event_bus import EventBus, get_event_bus
from core.events import (
    EventType, TickEvent, FeatureEvent, RegimeEvent, PatternEvent,
    PredictionEvent, ActionEvent, ExecutionEvent, FeedbackEvent
)
from features.pipeline import FeaturePipeline
from models.base import ModelBundle
from models.ensemble import EnsembleDecisionEngine
from models.ensemble import EnsembleDecisionEngine
from monitoring.metrics import MetricTracker
from regime.gmm import GaussianMixtureRegimeModel
from regime.hmm import MarkovTransitionModel
from memory.encoder import PatternEncoder
from memory.pattern_db import PatternDB
from learning.hidden_state import HiddenStateUpdater
from dataclasses import dataclass, field
from learning.online  import OnlineLearner
from risk.engine import RiskEngine
from data.collectors import SyntheticMarketDataCollector
from storage.warehouse import ReplayStore, StateStore

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Async feature extraction engine."""
    raw_events: list[dict[str, object]] = field(init=False, default_factory=list)
    feature_rows: list[dict[str, object]] = field(init=False, default_factory=list)
    
    def __init__(self, bus: EventBus, feature_pipeline: FeaturePipeline):
        self.bus = bus
        self.pipeline = feature_pipeline
        self.bus.subscribe(EventType.TICK, self.handle_tick)

    def _record_state(self, tick: TickEvent, state) -> None:
            self.raw_events.append(
                {
                    "timestamp": tick.timestamp,
                    "symbol": tick.symbol,
                    "price": tick.price,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "volume": tick.volume,
                    "spread": tick.spread,
                    "imbalance": tick.imbalance,
                    "metadata": tick.metadata,
                }
            )
            self.feature_rows.append(
                {
                    "timestamp": state.timestamp,
                    "symbol": state.symbol,
                    "timeframe": state.timeframe,
                    "filtered_price": state.filtered_price,
                    "volatility": state.volatility,
                    "features": state.features.tolist(),
                    "regime_probs": state.regime_probs.tolist(),
                    "hidden_state": state.hidden_state.tolist(),
                }
            )

    async def handle_tick(self, tick: TickEvent) -> None:
        """Process market tick and extract features."""
        try:
            state = self.pipeline.update(tick)
            self._record_state(tick, state)

            filtered_price, velocity = self.pipeline.get_state()
            
            feature_event = FeatureEvent(
                state=state
            )
            
            await self.bus.publish(feature_event)
        except Exception as e:
            logger.error(f"Error in FeatureEngine: {e}")


class RegimeEngine:
    """Async regime detection engine (GMM + HMM)."""
    
    def __init__(
        self,
        bus: EventBus,
        gmm: GaussianMixtureRegimeModel,
        hmm: MarkovTransitionModel
    ):
        self.bus = bus
        self.gmm = gmm
        self.hmm = hmm
        self.bus.subscribe(EventType.FEATURE, self.handle_features)
    
    async def handle_features(self, event: FeatureEvent) -> None:
        """Process features and detect regime."""
        try:
            # GMM: compute regime probabilities
            gmm_probs = self.gmm.update(event.state.features)
            
            # HMM: smooth probability with transition model
            hmm_probs = self.hmm.update(gmm_probs)
            
            regime_id = int(np.argmax(hmm_probs))
            
            regime_event = RegimeEvent(
                state=event.state,
                gmm_probs=gmm_probs,
                hmm_probs=hmm_probs,
                regime_id=regime_id,
                timestamp=datetime.now().timestamp(),
                metadata=None  # Optionally include metadata from the feature event,
            )
            
            await self.bus.publish(regime_event)
        except Exception as e:
            logger.error(f"Error in RegimeEngine: {e}")


class MemoryEngine:
    """Async pattern memory engine."""
    
    def __init__(
        self,
        bus: EventBus,
        pattern_encoder: PatternEncoder,
        pattern_db: PatternDB,
        hidden_state_updater: HiddenStateUpdater
    ):
        self.bus = bus
        self.encoder = pattern_encoder
        self.db = pattern_db
        self.hidden_updater = hidden_state_updater
        self.hidden_state = np.zeros(hidden_state_updater.hidden_dim)
        
        self.bus.subscribe(EventType.REGIME, self.handle_regime)
        self.bus.subscribe(EventType.FEATURE, self.current_features)
        
        self.last_features = None
    
    async def current_features(self, event: FeatureEvent) -> None:
        """Store features for later use."""
        self.last_features = event.state.features
       
    
    async def handle_regime(self, event: RegimeEvent) -> None:
        """Retrieve pattern from memory."""
        try:
            if self.last_features is None:
                return

            # Encode state to pattern key
            pattern_key = self.encoder.encode(
                self.last_features,
                event.hmm_probs
            )
            
            # Get pattern node
            pattern_node = self.db.get(pattern_key)
            pattern_record = self.db.get_or_create(
                self.last_features,
                self.hidden_state,
                event.hmm_probs
            )
            
            # Retrieve hidden state
            self.hidden_state = pattern_node.hidden_state.copy()
            
            pattern_event = PatternEvent(
                state=event.state,
                pattern_key=pattern_key,
                hidden_state=self.hidden_state,
                pattern=pattern_record,
                node_stats=pattern_node.get_stats(),
                timestamp=event.timestamp
            )
            
            await self.bus.publish(pattern_event)
        except Exception as e:
            logger.error(f"Error in MemoryEngine: {e}")


class PredictionEngine:
    """Async prediction engine."""
    
    def __init__(self, bus: EventBus, model_bundle: ModelBundle):
        self.bus = bus
        self.model_bundle = model_bundle

        self.bus.subscribe(EventType.PATTERN, self.handle_pattern)
        self.bus.subscribe(EventType.FEATURE, self.store_features)
        self.bus.subscribe(EventType.REGIME, self.store_regime)
        
        self.last_features = None
        self.last_regime_probs = None
        self.last_metadata = None
    
    async def store_features(self, event: FeatureEvent) -> None:
        self.last_features = event.state.features
        self.last_metadata = event.state.metadata
    
    async def store_regime(self, event: RegimeEvent) -> None:
        self.last_regime_probs = event.hmm_probs
    
    async def handle_pattern(self, event: PatternEvent) -> None:
        """Predict using model."""
        try:
            if self.last_features is None or self.last_regime_probs is None:
                return
            
            state = event.state
            
            # Concatenate features for prediction
            input_vec = np.concatenate([
                self.last_features,
                self.last_regime_probs,
                event.hidden_state
            ])

            model_output = self.model_bundle.predict(state.features) if self.model_bundle.fitted else self.model_bundle.predict(state.features)

            prediction_event = PredictionEvent(
                state=state,
                model_output=model_output,
                timestamp=event.timestamp,
                metadata=self.last_metadata,
            )
            
            await self.bus.publish(prediction_event)
        except Exception as e:
            logger.error(f"Error in PredictionEngine: {e}")


class DecisionEngine:
    """Async decision engine (MPC)."""
    
    def __init__(self, bus: EventBus, controller,ensemble: EnsembleDecisionEngine):
        self.bus = bus
        self.controller = controller
        self.ensemble = ensemble
        self.last_features = None
        self.bus.subscribe(EventType.PREDICTION, self.handle_prediction)
        self.bus.subscribe(EventType.FEATURE, self.store_features)
        self.bus.subscribe(EventType.REGIME, self.store_regime)
        self.bus.subscribe(EventType.PATTERN, self.store_features)

    async def store_features(self, event: FeatureEvent) -> None:
        self.last_features = event.state.features

    async def store_regime(self, event: RegimeEvent) -> None:
        self.last_regime_probs = event.hmm_probs

    async def store_pattern(self, event: PatternEvent) -> None:
        self.last_hidden_state = event.pattern

    async def handle_prediction(self, event: PredictionEvent) -> None:
        """Make decision using MPC."""
        try:
            state = event.state

            trajectory = self.controller.forecast(np.array([event.model_output.predicted_return] * self.controller.horizon), state.volatility, liquidity=1.0)

            # Decide action
            action_event = self.ensemble.decide(state, event.model_output, self.last_regime_probs, self.last_hidden_state, trajectory)

            await self.bus.publish(action_event)
        except Exception as e:
            logger.error(f"Error in DecisionEngine: {e}")


class ExecutionEngine:
    """Async execution engine."""
    
    def __init__(self, bus: EventBus, risk: RiskEngine, simulator):
        self.bus = bus
        self.risk = risk
        self.simulator = simulator
        self.last_prediction = 0.0
        self.prev_price: Optional[float] = None
        self.current_price: float = float(self.simulator.initial_price)
        self.bus.subscribe(EventType.ACTION, self.handle_action)
        self.bus.subscribe(EventType.PREDICTION, self.handle_prediction)
        # Keep simulator price updated on each feature tick for accurate PnL
        self.bus.subscribe(EventType.FEATURE, self.handle_feature)

    async def handle_prediction(self, event: PredictionEvent) -> None:
        """Store latest prediction to compute prediction error on execution."""
        try:
            self.last_prediction = float(event.model_output.predicted_return)
        except Exception:
            self.last_prediction = 0.0
    
    async def handle_action(self, event: ActionEvent) -> None:
        """Execute trade."""
        try:
            result = self.simulator.step(event.action)

            if self.prev_price is not None and self.prev_price != 0:
                actual_return = (self.current_price - self.prev_price) / abs(self.prev_price)
            else:
                actual_return = 0.0
            reward = actual_return

            sized_quantity = self.risk.size_position(
                portfolio_value=self.simulator.cash + self.simulator.position * event.state.filtered_price,
                confidence=event.confidence,
                volatility=event.state.volatility, liquidity=1.0)

            execution_event = ExecutionEvent(
                state=event.state,
                action=event.action,
                quantity=sized_quantity,
                price=self.current_price,
                transaction_cost=0.0,  
                position_after=self.simulator.position,
                cash_after=self.simulator.cash
            )
            
            await self.bus.publish(execution_event)

            feedback_event = FeedbackEvent(
                state=event.state,
                action=event.action,
                realized_pnl=reward,
                actual_return=actual_return,
                reward=reward,
                prediction_error=abs(actual_return - self.last_prediction)
            )
            await self.bus.publish(feedback_event)
        except Exception as e:
            logger.error(f"Error in ExecutionEngine: {e}")

    async def handle_feature(self, event: FeatureEvent) -> None:
        """Update simulator price on each feature tick."""
        try:
            price = float(event.state.filtered_price)
            self.prev_price = self.current_price
            self.current_price = price
            # update internal simulator price without executing a trade
            if hasattr(self.simulator, 'update_price'):
                self.simulator.update_price(price)
        except Exception as e:
            logger.debug(f"ExecutionEngine failed to update price: {e}")


class LearningEngine:
    """Async learning engine."""
    feedback_rows: list[dict[str, object]] = field(init=False, default_factory=list)
    
    
    def __init__(self, bus: EventBus, trainer, pattern_db: PatternDB,learner: OnlineLearner,metrics: MetricTracker):
        self.bus = bus
        self.trainer = trainer
        self.metrics = metrics
        self.pattern_db = pattern_db
        self.learner = learner
        self.last_features = None
        self.last_regime_probs = None
        self.last_prediction = 0.0
        self.last_hidden_state = None

        self.bus.subscribe(EventType.FEEDBACK, self.handle_feedback)
        self.bus.subscribe(EventType.FEATURE, self.handle_feature)
        self.bus.subscribe(EventType.REGIME, self.handle_regime)
        self.bus.subscribe(EventType.PREDICTION, self.handle_prediction)
        self.bus.subscribe(EventType.PATTERN, self.handle_pattern)
        self.bus.subscribe(EventType.EXECUTION, self.handle_execution)
        self.bus.subscribe(EventType.TICK, self.handle_ticker)

    async def handle_execution(self, event: ExecutionEvent) -> None:
        """Update metrics on execution."""
        try:
            self.portfolio_value = event.cash_after + event.position_after * event.price
        except Exception as e:
            logger.error(f"Error in LearningEngine during execution handling: {e}")

    async def handle_feature(self, event: FeatureEvent) -> None:
        self.last_features = event.state.features

    async def handle_ticker(self, event: TickEvent) -> None:
        self.tick = event

    async def handle_regime(self, event: RegimeEvent) -> None:
        self.last_regime_probs = event.hmm_probs

    async def handle_prediction(self, event: PredictionEvent) -> None:
        self.last_prediction = float(event.model_output.predicted_return)

    async def handle_pattern(self, event: PatternEvent) -> None:
        self.last_hidden_state = event.hidden_state
    
    async def handle_feedback(self, event: FeedbackEvent) -> None:
        """Learn from feedback."""
        try:
            tick = event.state
            
            self.learner.update(event)
            self.metrics.update_trade(event.reward, self.portfolio_value)
            self.feedback_rows.append(
                            {
                                "timestamp": event.state.timestamp,
                                "symbol": event.state.symbol,
                                "action": event.action.value,
                                "realized_pnl": event.realized_pnl,
                                "prediction_error": event.prediction_error,
                                "reward": event.reward,
                            }
                        )
            train_features = self.last_features
            if (
                self.last_features is not None
                and self.last_regime_probs is not None
                and self.last_hidden_state is not None
            ):
                train_features = np.concatenate([
                    self.last_features,
                    self.last_regime_probs,
                    self.last_hidden_state,
                ])

            experience = {
                'features': train_features,
                'regime_probs': self.last_regime_probs,
                'prediction': self.last_prediction,
                'hidden_state': self.last_hidden_state,
                'actual_return': event.actual_return,
                'prediction_error': event.prediction_error,
                'reward': event.reward
            }
            
            self.trainer.update(experience)
            
            # Optionally prune pattern DB if getting too large
            if self.pattern_db.get_size() > 1000:
                self.pattern_db.prune_least_frequent(keep_ratio=0.9)

            predicted_direction = 1 if self.last_prediction > 0 else -1 if self.last_prediction < 0 else 0
            actual_direction = 1 if self.tick.price > event.state.filtered_price else -1 if self.tick.price < event.state.filtered_price else 0
            self.metrics.update_classification(actual_direction, predicted_direction)


        except Exception as e:
            logger.error(f"Error in LearningEngine: {e}")


class AsyncPipeline:
    """
    Main async pipeline orchestrator.
    Coordinates all engines through event bus.
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        self.bus = bus or get_event_bus()
        self.engines = {}
        self.running = False
    
    def register_feature_engine(self, engine: FeatureEngine) -> None:
        """Register feature extraction engine."""
        self.engines['features'] = engine

    def register_regime_engine(self, engine: RegimeEngine) -> None:
        """Register regime detection engine."""
        self.engines['regime'] = engine
    
    def register_memory_engine(self, engine: MemoryEngine) -> None:
        """Register pattern memory engine."""
        self.engines['memory'] = engine
    
    def register_prediction_engine(self, engine: PredictionEngine) -> None:
        """Register prediction engine."""
        self.engines['prediction'] = engine
    
    def register_decision_engine(self, engine: DecisionEngine) -> None:
        """Register decision engine."""
        self.engines['decision'] = engine
    
    def register_execution_engine(self, engine: ExecutionEngine) -> None:
        """Register execution engine."""
        self.engines['execution'] = engine
    
    def register_learning_engine(self, engine: LearningEngine) -> None:
        """Register learning engine."""
        self.engines['learning'] = engine
    
    async def run(self, tick_stream) -> None:
        """
        Run pipeline.
        
        Args:
            tick_stream: Async generator yielding TickEvent objects
        """
        self.running = True
        logger.info("Starting async pipeline")
        
        try:
            async for tick_event in tick_stream:
                if not self.running:
                    break
                
                # Publish tick event
                await self.bus.publish(tick_event)
                
                # Small delay to allow processing
                await asyncio.sleep(0.001)
        
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
        finally:
            self.running = False
            logger.info("Pipeline stopped")
    
    def stop(self) -> None:
        """Stop pipeline."""
        self.running = False
    
    async def process_tick(
        self,
        price: float,
        timestamp: float,
        bid: float | None = None,
        ask: float | None = None,
        volume: float | None = None,
        spread: float | None = None,
        imbalance: float | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process a single market tick."""
        tick_event = TickEvent(
            symbol=self.engines['features'].pipeline.symbol,
            price=price,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            volume=volume,
            spread=spread,
            imbalance=imbalance,
            metadata=metadata # type: ignore
        )
        await self.bus.publish(tick_event)

   
    
   