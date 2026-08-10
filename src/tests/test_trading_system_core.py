from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from app import TradingIntelligenceSystem
from core.event_bus import EventBus
from core.events import TickEvent, TradeAction
from data.collectors import OptionChainCollector, VolatilityCollector
from trading_system.execution.engine import ExecutionEngine
from models.base import ModelBundle
from features.pipeline import FeaturePipeline
from trading_system.memory.patterns import PatternDB
from replay.engine import ReplayEngine
from storage.warehouse import AnalyticsStore


def test_event_bus_publishes_in_order() -> None:
    bus = EventBus()
    observed: list[int] = []

    async def handler(event: TickEvent) -> None:
        observed.append(int(event.price))

    bus.subscribe(TickEvent, handler)

    async def run() -> None:
        await bus.publish(TickEvent(timestamp=1.0, symbol="TEST", price=1.0))
        await bus.publish(TickEvent(timestamp=2.0, symbol="TEST", price=2.0))

    asyncio.run(run())
    assert observed == [1, 2]


def test_feature_pipeline_builds_state_vector() -> None:
    pipeline = FeaturePipeline(symbol="TEST")
    state = pipeline.update(TickEvent(timestamp=1.0, symbol="TEST", price=100.0, volume=10.0, bid=99.5, ask=100.5))
    assert state.symbol == "TEST"
    assert state.features.shape[0] >= 10
    assert np.isfinite(state.features).all()


def test_pattern_db_retrieval_and_branching() -> None:
    db = PatternDB()
    feature_snapshot = np.array([1.0, 0.5, -0.2])
    hidden_state = np.array([0.1, 0.2, 0.3])
    regime_probs = np.array([0.6, 0.3, 0.1])
    record = db.get_or_create(feature_snapshot, hidden_state, regime_probs)
    retrieved = db.retrieve(feature_snapshot)
    assert retrieved[0].pattern_id == record.pattern_id
    branched = db.branch(record, feature_snapshot + 0.01, hidden_state, regime_probs)
    assert branched.parent_pattern_id == record.pattern_id


def test_execution_engine_state_machine() -> None:
    pipeline = FeaturePipeline(symbol="TEST")
    state = pipeline.update(TickEvent(timestamp=1.0, symbol="TEST", price=100.0))
    engine = ExecutionEngine(initial_cash=100_000.0)

    buy_event = engine.step(
        type("ActionLike", (), {"state": state, "action": TradeAction.BUY, "confidence": 0.9, "expected_reward": 1.0, "expected_risk": 0.1})()
    )
    assert buy_event[0].position_after >= 0.0


def test_replay_engine_streams_frame() -> None:
    frame = pd.DataFrame({"timestamp": [1.0, 2.0], "Close": [100.0, 101.0]})
    engine = ReplayEngine(output_path=__import__("pathlib").Path("results/test_replay"), delay_per_tick=0.0)

    async def collect() -> list[float]:
        values: list[float] = []
        async for tick in engine.from_frame(frame, symbol="TEST"):
            values.append(tick.price)
        return values

    assert asyncio.run(collect()) == [100.0, 101.0]


def test_model_bundle_online_update_enables_prediction() -> None:
    model = ModelBundle(feature_dim=4)
    model.update(np.array([0.1, 0.2, 0.3, 0.4]), action_label=1, realized_return=0.05)
    output = model.predict(np.array([0.1, 0.2, 0.3, 0.4]))
    assert model.online_ready is True
    assert set(output.action_logits) == {"BUY", "SELL", "HOLD"}


def test_system_can_run_small_synthetic_cycle() -> None:
    system = TradingIntelligenceSystem()

    async def run() -> dict[str, dict[str, float]]:
        await system.run_synthetic(steps=5)
        return system.report()

    report = asyncio.run(run())
    assert "classification" in report
    assert "trading" in report


def test_collectors_and_storage_write_outputs(tmp_path: Path) -> None:
    async def collect_option_and_vol() -> tuple[list[float], list[float]]:
        option_prices: list[float] = []
        vol_prices: list[float] = []

        async for tick in OptionChainCollector(symbol="TEST", delay_per_tick=0.0).stream():
            option_prices.append(tick.price)
            if len(option_prices) >= 3:
                break

        async for tick in VolatilityCollector(symbol="TEST", delay_per_tick=0.0).stream():
            vol_prices.append(tick.price)
            if len(vol_prices) >= 3:
                break

        return option_prices, vol_prices

    option_prices, vol_prices = asyncio.run(collect_option_and_vol())
    assert len(option_prices) == 3
    assert len(vol_prices) == 3

    store = AnalyticsStore(tmp_path)
    frame_path = store.put_frame("features", pd.DataFrame({"a": [1, 2, 3]}))
    events_path = store.save_events("raw", [{"timestamp": 1.0, "symbol": "TEST", "price": 100.0}])
    metrics_path = store.write_json("metrics", {"ok": True})

    assert frame_path.exists()
    assert events_path.exists()
    assert metrics_path.exists()

    system = TradingIntelligenceSystem()

    async def run_and_flush() -> dict[str, dict[str, float]]:
        await system.run_synthetic(steps=3)
        outputs = system.flush_storage()
        return system.report(), outputs

    report, outputs = asyncio.run(run_and_flush())
    assert "classification" in report
    assert outputs["metrics"].exists()
    assert outputs["replay_log"].exists()
