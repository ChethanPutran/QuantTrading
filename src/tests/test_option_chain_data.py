import asyncio
import sys
import types

import numpy as np
import pandas as pd

from data.market_data.option_chain import get_index_option_chain_data
from pipelines.async_pipeline import FeatureEngine
from smart.trading_system.comm.events import TickEvent


class FakeOptionTicker:
    def __init__(self):
        self.options = ["2026-05-07"]
        self.fast_info = {"lastPrice": 100.0}

    def option_chain(self, expiry):
        assert expiry == "2026-05-07"
        calls = pd.DataFrame(
            [
                {
                    "contractSymbol": "NIFTY260507C00100000",
                    "strike": 100.0,
                    "lastPrice": 2.5,
                    "bid": 2.4,
                    "ask": 2.6,
                    "volume": 100,
                    "openInterest": 250,
                    "inTheMoney": False,
                }
            ]
        )
        puts = pd.DataFrame(
            [
                {
                    "contractSymbol": "NIFTY260507P00100000",
                    "strike": 100.0,
                    "lastPrice": 2.2,
                    "bid": 2.1,
                    "ask": 2.3,
                    "volume": 80,
                    "openInterest": 180,
                    "inTheMoney": False,
                }
            ]
        )
        return types.SimpleNamespace(calls=calls, puts=puts)


class DummyPipeline:
    def update(self, price):
        return np.array([price, price * 0.01])

    def get_state(self):
        return 99.5, 0.25


class DummyBus:
    def __init__(self):
        self.published = []

    def subscribe(self, *args, **kwargs):
        return None

    async def publish(self, event):
        self.published.append(event)


def test_get_index_option_chain_data_normalizes_payload(monkeypatch):
    fake_module = types.SimpleNamespace(Ticker=lambda symbol: FakeOptionTicker())
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)

    snapshot = get_index_option_chain_data("^NSEI")

    assert snapshot["symbol"] == "^NSEI"
    assert snapshot["expiry"] == "2026-05-07"
    assert snapshot["summary"]["calls_count"] == 1
    assert snapshot["summary"]["puts_count"] == 1
    assert snapshot["summary"]["atm_strike"] == 100.0
    assert snapshot["calls"][0]["side"] == "call"
    assert snapshot["puts"][0]["side"] == "put"


def test_feature_engine_propagates_tick_metadata():
    bus = DummyBus()
    engine = FeatureEngine(bus, DummyPipeline())
    metadata = {"summary": {"calls_count": 1}}

    asyncio.run(
        engine.handle_tick(
            TickEvent(price=100.0, timestamp=0.0, metadata=metadata)
        )
    )

    assert len(bus.published) == 1
    assert bus.published[0].metadata == metadata