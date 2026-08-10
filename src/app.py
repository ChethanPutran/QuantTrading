"""
Main entry point for the trading system.
Integrates all modules in async event-driven architecture.
"""

import argparse
import asyncio
import logging
from anyio import Path
import numpy as np
from datetime import datetime
from typing import AsyncGenerator, Optional

import pandas as pd

from config.settings import AppSettings, FeaturePipelineSettings, MPCSettings, PortfolioSettings, RegimeModelSettings, RuntimeSettings
from data.collectors import SyntheticMarketDataCollector
from learning.online import OnlineLearner
from models.base import ModelBundle
from models.ensemble import EnsembleDecisionEngine
from monitoring.metrics import MetricTracker
from monitoring.metrics import MetricTracker
from risk.engine import RiskEngine
from storage.warehouse import AnalyticsStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system components
from core.event_bus import init_event_bus, get_event_bus
from utils.latency_monitor import init_latency_tracker, ModuleType, get_latency_tracker
from core.events import FeedbackEvent, TickEvent

from features.pipeline import FeaturePipeline
from filters.kalman import KalmanFilter

from data.market_data.option_chain import (
    get_index_option_chain_data,
    get_index_option_chain_history,
)

from regime.gmm import GaussianMixtureRegimeModel, OnlineGMMRegimeModel
from regime.hmm import MarkovTransitionModel, OnlineHMMRegimeModel
from models.linear import LinearModel

from memory.encoder import PatternEncoder
from memory.pattern_db import PatternDB

from learning.hidden_state import HiddenStateUpdater
from learning.trainer import OnlineTrainer

from control.mpc import MPCController

from execution.simulator import TradingSimulator

from pipelines.async_pipeline import (
    AsyncPipeline, FeatureEngine, RegimeEngine, MemoryEngine,
    PredictionEngine, DecisionEngine, ExecutionEngine, LearningEngine
)


class TradingSystem:
    """
    Main trading system orchestrator.
    Integrates all modules with async event-driven architecture.
    """
    
    def __init__(
        self,
        settings: AppSettings
   
    ):
        """
        Initialize trading system.
        
        Args:
            settings: Application settings/configuration

        """
        self.settings = settings
        
        # Initialize event bus
        self.bus = init_event_bus(max_queue_size=100)
        
        # Initialize latency tracker
        if settings.runtime.enable_latency_tracking:
            self.tracker = init_latency_tracker(enable_alerts=False)
        else:
            self.tracker = None
        
        # Initialize components
        logger.info("Initializing trading system components...")
        
        # Feature extraction
        self.feature_pipeline = FeaturePipeline(
            config=settings.feature_pipeline
        )

        
        # Regime detection
        self.gmm = GaussianMixtureRegimeModel(
            n_regimes=settings.regime_model.n_regimes,
            feature_dim=settings.feature_pipeline.feature_dim,
            learning_rate=0.01,
            covariance_type='diag'
        )
        # self.gmm = OnlineGMMRegimeModel(feature_dim=settings.feature_pipeline.feature_dim, n_regimes=settings.regime_model.n_regimes)
        
        self.hmm = MarkovTransitionModel(
            n_regimes=settings.regime_model.n_regimes,
            smoothing=0.7
        )
        # self.hmm = OnlineHMMRegimeModel(n_regimes=self.gmm.n_regimes)
        
        # Memory system
        self.pattern_encoder = PatternEncoder(
            feature_bin_size=0.1,
            regime_dim=settings.regime_model.n_regimes,
        )
        
        self.pattern_db = PatternDB(max_patterns=1000)
        
        # Hidden state
        self.hidden_state_updater = HiddenStateUpdater(
            hidden_dim=8,
            decay=0.95,
            learning_rate=0.01
        )
        
        # Prediction model
        self.linear_model = LinearModel(
            n_features=
            settings.feature_pipeline.feature_dim + settings.regime_model.n_regimes + 8,
            learning_rate=0.01,
            l2=0.001,
            fit_intercept=True
        )
    
                
        
        # Control
        self.mpc_controller = MPCController(
            model=self.linear_model,
            horizon=settings.mpc.mpc_horizon,
            action_space=[-1, 0, 1],
            max_position=1.0,
            cost_per_trade=0.001,
            risk_penalty=0.1,
            use_pruning=True
        )

        self.model_bundle = ModelBundle(feature_dim=self.settings.feature_pipeline.feature_dim)
        self.ensemble = EnsembleDecisionEngine()
        self.learner = OnlineLearner(self.model_bundle, self.pattern_db)
        self.metrics = MetricTracker()
        self.trainer = OnlineTrainer(
            model=self.linear_model,
            hidden_state_updater=self.hidden_state_updater,
            pattern_db=self.pattern_db,
            pattern_encoder=self.pattern_encoder,
            error_threshold=0.5,
            branching_enabled=True
        )
        
        # Execution
        self.simulator = TradingSimulator(
            initial_balance=settings.portfolio.initial_cash,
            initial_price=settings.portfolio.initial_price,
            max_position=settings.portfolio.max_position,
            transaction_cost=settings.portfolio.transaction_cost,
            slippage=settings.portfolio.slippage
        )

        self.risk = RiskEngine(
                    max_position_fraction=settings.risk.max_position_fraction,
                    max_drawdown_fraction=settings.risk.max_drawdown_fraction,
                    confidence_floor=settings.risk.min_confidence,
                    liquidity_penalty=settings.risk.liquidity_penalty,
                )

        self.metrics = MetricTracker()
        self.state_store = AnalyticsStore(self.settings.runtime.state_store_path, redis_url=self.settings.runtime.redis_url)
        self.replay_store = AnalyticsStore(self.settings.runtime.replay_store_path, redis_url=self.settings.runtime.redis_url)

        
        # Initialize async pipeline
        self.pipeline = AsyncPipeline(bus=self.bus)
        
        # Create engines
        self._initialize_engines()
        
        logger.info("Trading system initialized successfully")
    
    def _initialize_engines(self) -> None:
        """Initialize all async engines."""
        logger.info("Initializing async engines...")
        
        # Feature engine
        feature_engine = FeatureEngine(self.bus, self.feature_pipeline)
        self.pipeline.register_feature_engine(feature_engine)
        
        # Regime engine
        regime_engine = RegimeEngine(self.bus, self.gmm, self.hmm)
        self.pipeline.register_regime_engine(regime_engine)
        
        # Memory engine
        memory_engine = MemoryEngine(
            self.bus,
            self.pattern_encoder,
            self.pattern_db,
            self.hidden_state_updater
        )
        self.pipeline.register_memory_engine(memory_engine)
        
        # Prediction engine
        prediction_engine = PredictionEngine(self.bus, self.model_bundle)
        self.pipeline.register_prediction_engine(prediction_engine)
        
        # Decision engine
        decision_engine = DecisionEngine(self.bus, self.mpc_controller, self.ensemble)
        self.pipeline.register_decision_engine(decision_engine)
        
        # Execution engine
        execution_engine = ExecutionEngine(self.bus,self.risk, self.simulator)
        self.pipeline.register_execution_engine(execution_engine)
        
        # Learning engine
        learning_engine = LearningEngine(self.bus, self.trainer, self.pattern_db, self.learner, self.metrics)
        self.pipeline.register_learning_engine(learning_engine)
    
    async def market_data_stream(
        self,
        price_data: Optional[list] = None,
        ticker: Optional[str] = None,
        csv_path: Optional[str] = None,
        index_symbol: Optional[str] = None,
        option_chain_expiry: Optional[str] = None,
        history: bool = False,
        history_days: int = 30,
        history_interval: str = "1d",
        delay_per_tick: float = 0.1,
    ) -> AsyncGenerator:
        """
        Generate market data stream.
        
        Args:
            price_data: List of prices (if provided, streamed directly)
            ticker: Ticker symbol to fetch via `data.market_data` (optional)
            csv_path: Path to CSV with OHLCV data (optional)
            delay_per_tick: Delay between ticks in seconds

        Yields:
            TickEvent objects (uses 'Close' column for OHLCV frames)
        """
        from core.events import TickEvent

        # Determine source of price data
        prices = None
        option_chain_context = None
        historical_snapshots = None

        if history and index_symbol is not None:
            try:
                historical_snapshots = get_index_option_chain_history(
                    index_symbol,
                    expiry=option_chain_expiry,
                    days=history_days,
                    interval=history_interval,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch option chain history for %s: %s",
                    index_symbol,
                    exc,
                )

        elif index_symbol is not None:
            try:
                option_chain_context = get_index_option_chain_data(
                    index_symbol,
                    expiry=option_chain_expiry,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch option chain for %s: %s",
                    index_symbol,
                    exc,
                )

        if price_data is not None:
            prices = list(price_data)

        elif csv_path is not None:
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if 'Close' in df.columns:
                    prices = df['Close'].astype(float).tolist()
                elif 'close' in df.columns:
                    prices = df['close'].astype(float).tolist()
                else:
                    # fallback to first numeric column
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    prices = df[numeric_cols[0]].astype(float).tolist()
            except Exception as e:
                logger.error(f"Failed to read csv_path={csv_path}: {e}")
                prices = []

        elif ticker is not None:
            try:
                from data.market_data.market_data import load_stock_data

                status, df = load_stock_data(ticker, refresh=False)
                if status and not df.empty:
                    # Use Close column if available
                    if 'Close' in df.columns:
                        prices = df['Close'].astype(float).tolist()
                    elif 'close' in df.columns:
                        prices = df['close'].astype(float).tolist()
                    else:
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        prices = df[numeric_cols[0]].astype(float).tolist()
                else:
                    logger.warning(f"No data found for ticker={ticker}")
                    prices = []
            except Exception as e:
                logger.error(f"Error fetching ticker {ticker}: {e}")
                prices = []

        else:
            prices = []

        if historical_snapshots is not None:
            for snapshot in historical_snapshots:
                price = snapshot.get("underlying_price")
                if price is None:
                    continue

                timestamp = snapshot.get("date")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp).timestamp()
                    except ValueError:
                        timestamp = datetime.now().timestamp()
                elif timestamp is None:
                    timestamp = datetime.now().timestamp()

                yield TickEvent(
                    symbol=self.settings.portfolio.index_symbol,
                    price=float(price),
                    timestamp=float(timestamp),
                    metadata=snapshot or {},
                )

                await asyncio.sleep(delay_per_tick)

            return

        # Stream the resolved prices
        for i, price in enumerate(prices):
            timestamp = datetime.now().timestamp()

            yield TickEvent(
                symbol=self.settings.portfolio.index_symbol,
                price=float(price),
                timestamp=timestamp,
                metadata=option_chain_context or {},
            )

            # Simulate real-time delays
            await asyncio.sleep(delay_per_tick)
    
    async def run(
        self,
        price_data: Optional[list] = None,
        ticker: Optional[str] = None,
        csv_path: Optional[str] = None,
        index_symbol: Optional[str] = None,
        option_chain_expiry: Optional[str] = None,
        history: bool = False,
        history_days: int = 30,
        history_interval: str = "1d",
        delay_per_tick: float = 0.01,
    ) -> None:
        """
        Run the trading system.
        
        Args:
            price_data: List of prices to process
            delay_per_tick: Delay between ticks
        """
        if price_data is not None:
            n_ticks = len(price_data)
        else:
            n_ticks = 'unknown'

        logger.info(f"Starting trading system with {n_ticks} ticks")

        tick_stream = self.market_data_stream(
            price_data=price_data,
            ticker=ticker,
            csv_path=csv_path,
            index_symbol=index_symbol,
            option_chain_expiry=option_chain_expiry,
            history=history,
            history_days=history_days,
            history_interval=history_interval,
            delay_per_tick=delay_per_tick,
        )
        
        await self.pipeline.run(tick_stream)
        
        logger.info("Trading system run completed")

    async def run_train_then_simulate(
        self,
        index_symbol: str,
        option_chain_expiry: Optional[str] = None,
        train_days: int = 180,
        simulate_days: int = 30,
        history_interval: str = "1d",
        delay_per_tick: float = 0.0,
    ) -> None:
        """Train on older history, then simulate on the most recent month.

        The model/memory state is retained between phases. Simulator state is
        reset before simulation so reported PnL is measured on the simulation split.
        """
        from core.events import TickEvent

        total_days = max(1, train_days + simulate_days)
        snapshots = get_index_option_chain_history(
            index_symbol,
            expiry=option_chain_expiry,
            days=total_days,
            interval=history_interval,
        )
        if len(snapshots) < 2:
            raise ValueError(
                f"Not enough history to split train/simulate for {index_symbol}."
            )

        # Split chronologically: oldest -> newest
        split_index = max(1, len(snapshots) - simulate_days)
        train_snapshots = snapshots[:split_index]
        simulate_snapshots = snapshots[split_index:]

        logger.info(
            "Phase 1 training on %d ticks, phase 2 simulation on %d ticks",
            len(train_snapshots),
            len(simulate_snapshots),
        )

        async def snapshot_stream(phase_snapshots: list[dict], phase: str):
            for snap in phase_snapshots:
                price = snap.get("underlying_price")
                if price is None:
                    continue

                timestamp = snap.get("date")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp).timestamp()
                    except ValueError:
                        timestamp = datetime.now().timestamp()
                elif timestamp is None:
                    timestamp = datetime.now().timestamp()

                metadata = dict(snap)
                metadata["phase"] = phase
                yield TickEvent(
                    symbol=self.settings.portfolio.index_symbol,
                    price=float(price),
                    timestamp=float(timestamp),
                    metadata=metadata,
                )
                await asyncio.sleep(delay_per_tick)

        # Training pass (updates model/pattern state)
        await self.pipeline.run(snapshot_stream(train_snapshots, phase="train"))

        # Reset execution account before simulation while retaining learned model state
        self.simulator.reset()

        # Simulation pass (evaluation window)
        await self.pipeline.run(snapshot_stream(simulate_snapshots, phase="simulate"))

        logger.info("Train+simulate run completed")
    
    def process_tick(
        self,
        price: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Process single market tick (synchronous wrapper).
        
        Args:
            price: Current market price
        """
        from core.events import TickEvent
        
        tick_event = TickEvent(
            symbol=self.settings.portfolio.index_symbol,
            price=price,
            timestamp=datetime.now().timestamp(),
            metadata=metadata or {},
        )
        
        # Process asynchronously if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.bus.publish(tick_event))
        except RuntimeError:
            # No running loop, create one
            asyncio.run(self.bus.publish(tick_event))
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        stats = {
            'simulator': self.simulator.get_statistics(),
            'trainer': self.trainer.get_statistics(),
            'mpc': self.mpc_controller.get_stats(),
            'pattern_db': self.pattern_db.get_stats(),
            'bus': self.bus.get_stats()
        }
        
        if self.tracker:
            stats['latency'] = self.tracker.get_statistics()
        
        return stats

    async def run_replay(self, frame: pd.DataFrame, symbol: str | None = None) -> None:
        symbol = symbol or self.settings.replay.symbol
        for _, row in frame.iterrows():
            # Convert pandas Series to dict for metadata to satisfy expected type
            tick = TickEvent(
                timestamp=float(row.get("timestamp", 0.0)),
                symbol=symbol,
                price=float(row.get("Close", row.iloc[0])),
                metadata=row.to_dict()
            )
            self.process_tick(
                price=tick.price,
                metadata=tick.metadata
            )
        self.flush_storage()
    
    async def run_synthetic(self, steps: int = 1000) -> None:
            collector = SyntheticMarketDataCollector(symbol=self.settings.replay.symbol, steps=steps, delay_per_tick=self.settings.replay.delay_per_tick, seed=self.settings.replay.deterministic_seed)
            async for tick in collector.stream():
                self.process_tick(
                    price=tick.price,
                    metadata=tick.metadata
                )
            self.flush_storage()

    def flush_storage(self):
        # """Flush analytics to storage."""
        # state_path = self.state_store.save()
        # replay_path = self.replay_store.save()
        # logger.info("Flushed analytics to storage: state=%s, replay=%s", state_path, replay_path)
        # return {"state": state_path, "replay": replay_path}
        pass
        
    def report(self) -> dict[str, dict[str, float]]:
        return {
            "classification": self.metrics.classification_summary(),
            "trading": self.metrics.trading_summary(),
        }
    
    def print_report(self) -> None:
        """Print system report."""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("TRADING SYSTEM REPORT")
        print("="*70)
        
        print("\n📊 SIMULATOR STATS:")
        for key, val in stats['simulator'].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        
        print("\n🧠 TRAINER STATS:")
        for key, val in stats['trainer'].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        
        print("\n🎯 MPC STATS:")
        for key, val in stats['mpc'].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        
        print("\n💾 PATTERN DB STATS:")
        for key, val in stats['pattern_db'].items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        
        print("\n🚌 EVENT BUS STATS:")
        for key, val in stats['bus'].items():
            print(f"  {key}: {val}")
        
        if self.tracker:
            print("\n⏱️  LATENCY REPORT:")
            self.tracker.print_report()
        
        print("="*70 + "\n")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the trading system or replay option-chain history.")
    parser.add_argument("--index-symbol", default=None, help="Index symbol to fetch option-chain data for, e.g. ^NSEI")
    parser.add_argument("--option-chain-expiry", default=None, help="Optional expiry date for the option chain")
    parser.add_argument("--ticker", default=None, help="Equity ticker to load historical OHLCV data for")
    parser.add_argument("--csv-path", default=None, help="Path to a CSV file containing historical prices or OHLCV data")
    parser.add_argument("--history", action="store_true", help="Replay a historical option-chain stream instead of a synthetic random walk")
    parser.add_argument("--history-days", type=int, default=30, help="Number of days to replay when --history is set")
    parser.add_argument("--history-interval", default="1d", help="Underlying price history interval used when --history is set")
    parser.add_argument("--train-simulate", action="store_true", help="Train on older history then simulate on the latest window")
    parser.add_argument("--train-days", type=int, default=180, help="Training window size in days for --train-simulate")
    parser.add_argument("--simulate-days", type=int, default=30, help="Simulation window size in days for --train-simulate")
    parser.add_argument("--delay-per-tick", type=float, default=0.001, help="Delay between replay ticks")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Starting cash balance")
    parser.add_argument("--initial-price", type=float, default=100.0, help="Starting price for synthetic runs")
    args = parser.parse_args()

    logger.info("Starting trading system...")

    if (
        args.index_symbol
        and not args.history
        and not args.train_simulate
        and args.ticker is None
        and args.csv_path is None
    ):
        # The documentation describes the default usage as historical replay,
        # so an index symbol should automatically activate the train/sim split.
        args.train_simulate = True
    
    # Initialize system
    system = TradingSystem(
        settings=AppSettings(
            portfolio=PortfolioSettings(
                initial_cash=args.initial_balance,
                initial_price=args.initial_price,
            ),
            feature_pipeline=FeaturePipelineSettings(feature_dim=8),
            regime_model=RegimeModelSettings(feature_dim=8, n_regimes=3),
            mpc=MPCSettings(mpc_horizon=5),
            runtime=RuntimeSettings(enable_latency_tracking=True)
        )
    )

    if args.train_simulate:
        if not args.index_symbol:
            raise SystemExit("--train-simulate requires --index-symbol")

        await system.run_train_then_simulate(
            index_symbol=args.index_symbol,
            option_chain_expiry=args.option_chain_expiry,
            train_days=args.train_days,
            simulate_days=args.simulate_days,
            history_interval=args.history_interval,
            delay_per_tick=args.delay_per_tick,
        )
    elif args.history:
        if not args.index_symbol:
            raise SystemExit("--history requires --index-symbol")

        await system.run(
            index_symbol=args.index_symbol,
            option_chain_expiry=args.option_chain_expiry,
            history=True,
            history_days=args.history_days,
            history_interval=args.history_interval,
            delay_per_tick=args.delay_per_tick,
        )
    elif args.ticker is not None or args.csv_path is not None:
        await system.run(
            ticker=args.ticker,
            csv_path=args.csv_path,
            delay_per_tick=args.delay_per_tick,
        )
    else:
        # Generate sample price data (random walk)
        np.random.seed(42)
        prices = [args.initial_price]
        for _ in range(1000):
            price_change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + price_change))

        # Run system
        await system.run(prices, delay_per_tick=args.delay_per_tick)
    
    # Print report
    system.print_report()


if __name__ == "__main__":
    asyncio.run(main())
