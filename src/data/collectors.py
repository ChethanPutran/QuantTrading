"""Async market data collectors with deterministic replay-friendly fallbacks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Sequence

import numpy as np
import pandas as pd

from ....src.trading_system.core.events import TickEvent
from ....src.trading_system.data.schemas import OptionChainSnapshot, TickSnapshot


@dataclass
class BaseCollector:
    symbol: str
    delay_per_tick: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    async def stream(self) -> AsyncGenerator[TickEvent, None]:
        raise NotImplementedError


def _normalize_timestamp(value: Any, fallback_index: int = 0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return datetime.now(timezone.utc).timestamp() + fallback_index
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return datetime.now(timezone.utc).timestamp() + fallback_index
    return datetime.now(timezone.utc).timestamp() + fallback_index


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except Exception:
        return None


@dataclass
class SyntheticMarketDataCollector(BaseCollector):
    start_price: float = 100.0
    steps: int = 1000
    drift: float = 0.0002
    volatility: float = 0.01
    seed: int = 42

    async def stream(self) -> AsyncGenerator[TickEvent, None]:
        rng = np.random.default_rng(self.seed)
        price = float(self.start_price)
        for index in range(self.steps):
            shock = rng.normal(self.drift, self.volatility)
            price = max(0.01, price * (1.0 + shock))
            timestamp = datetime.now(timezone.utc).timestamp() + index
            metadata = dict(self.metadata)
            metadata.update({"source": "synthetic", "sequence": index})
            yield TickEvent(timestamp=timestamp, symbol=self.symbol, price=price, metadata=metadata)
            if self.delay_per_tick > 0:
                await asyncio.sleep(self.delay_per_tick)


@dataclass
class CSVMarketDataCollector(BaseCollector):
    path: Path = field(default_factory=Path)
    price_column: str = "Close"
    timestamp_column: str = "timestamp"

    async def stream(self) -> AsyncGenerator[TickEvent, None]:
        frame = pd.read_csv(self.path)
        if self.price_column not in frame.columns:
            numeric_columns = frame.select_dtypes(include=["number"]).columns
            if len(numeric_columns) == 0:
                raise ValueError(f"No numeric price column found in {self.path}")
            self.price_column = str(numeric_columns[0])

        for index, row in frame.iterrows():
            timestamp = _normalize_timestamp(row.get(self.timestamp_column), index)
            yield TickEvent(
                timestamp=timestamp,
                symbol=self.symbol,
                price=float(row[self.price_column]),
                volume=_safe_float(row["Volume"]) if "Volume" in row else None,
                bid=_safe_float(row["Bid"]) if "Bid" in row else None,
                ask=_safe_float(row["Ask"]) if "Ask" in row else None,
                metadata={**row.to_dict(), "source": "csv"},
            )
            if self.delay_per_tick > 0:
                await asyncio.sleep(self.delay_per_tick)


@dataclass
class OptionChainCollector(BaseCollector):
    path: Path | None = None
    expiry: str | None = None
    synthetic_underlying_price: float = 100.0
    rows: Iterable[dict[str, Any]] | None = None

    def _load_rows(self) -> list[dict[str, Any]]:
        if self.rows is not None:
            return [dict(row) for row in self.rows]
        if self.path is not None and self.path.exists():
            frame = pd.read_csv(self.path)
            return frame.to_dict(orient="records")
        return []

    async def stream(self) -> AsyncGenerator[TickEvent, None]:
        rows = self._load_rows()
        if not rows:
            rng = np.random.default_rng(42)
            underlying = float(self.synthetic_underlying_price)
            for index in range(128):
                underlying *= 1.0 + rng.normal(0.0001, 0.005)
                implied_vol = abs(rng.normal(0.18, 0.04))
                snapshot = OptionChainSnapshot(
                    timestamp=datetime.now(timezone.utc).timestamp() + index,
                    symbol=self.symbol,
                    expiry=self.expiry,
                    underlying_price=underlying,
                    implied_volatility=implied_vol,
                    open_interest=float(max(0.0, rng.normal(250_000, 50_000))),
                    gamma_exposure=float(rng.normal(0.0, 1.0)),
                    call_put_imbalance=float(rng.normal(0.0, 0.2)),
                    iv_skew=float(rng.normal(0.0, 0.05)),
                    metadata={"source": "synthetic_option_chain"},
                )
                yield TickEvent(
                    timestamp=snapshot.timestamp,
                    symbol=self.symbol,
                    price=snapshot.underlying_price,
                    metadata={"option_chain": asdict(snapshot)},
                )
                if self.delay_per_tick > 0:
                    await asyncio.sleep(self.delay_per_tick)
            return

        for index, row in enumerate(rows):
            timestamp = _normalize_timestamp(row.get("timestamp"), index)
            underlying = _safe_float(row.get("underlying_price", row.get("price", self.synthetic_underlying_price))) or self.synthetic_underlying_price
            snapshot = OptionChainSnapshot(
                timestamp=timestamp,
                symbol=self.symbol,
                expiry=row.get("expiry", self.expiry),
                underlying_price=underlying,
                implied_volatility=_safe_float(row.get("implied_volatility")),
                open_interest=_safe_float(row.get("open_interest")),
                gamma_exposure=_safe_float(row.get("gamma_exposure")),
                call_put_imbalance=_safe_float(row.get("call_put_imbalance")),
                iv_skew=_safe_float(row.get("iv_skew")),
                metadata={"source": row.get("source", "option_chain")},
            )
            yield TickEvent(timestamp=timestamp, symbol=self.symbol, price=underlying, metadata={"option_chain": asdict(snapshot)})
            if self.delay_per_tick > 0:
                await asyncio.sleep(self.delay_per_tick)


@dataclass
class VolatilityCollector(BaseCollector):
    path: Path | None = None
    rows: Iterable[dict[str, Any]] | None = None
    window: int = 20

    def _load_rows(self) -> list[dict[str, Any]]:
        if self.rows is not None:
            return [dict(row) for row in self.rows]
        if self.path is not None and self.path.exists():
            frame = pd.read_csv(self.path)
            return frame.to_dict(orient="records")
        return []

    async def stream(self) -> AsyncGenerator[TickEvent, None]:
        rows = self._load_rows()
        if not rows:
            rng = np.random.default_rng(7)
            realized = 0.0
            implied = 0.2
            for index in range(128):
                realized = max(0.0, 0.9 * realized + abs(rng.normal(0.0, 0.02)))
                implied = max(0.01, 0.95 * implied + 0.05 * abs(rng.normal(0.2, 0.03)))
                timestamp = datetime.now(timezone.utc).timestamp() + index
                yield TickEvent(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    price=implied,
                    metadata={
                        "source": "synthetic_volatility",
                        "realized_volatility": realized,
                        "implied_volatility": implied,
                        "volatility_regime": "high" if implied > 0.25 else "calm",
                    },
                )
                if self.delay_per_tick > 0:
                    await asyncio.sleep(self.delay_per_tick)
            return

        for index, row in enumerate(rows):
            timestamp = _normalize_timestamp(row.get("timestamp"), index)
            realized = _safe_float(row.get("realized_volatility"))
            implied = _safe_float(row.get("implied_volatility"))
            yield TickEvent(
                timestamp=timestamp,
                symbol=self.symbol,
                price=float(implied if implied is not None else realized if realized is not None else 0.0),
                metadata={
                    "source": row.get("source", "volatility"),
                    "realized_volatility": realized,
                    "implied_volatility": implied,
                    "volatility_regime": row.get("volatility_regime", "unknown"),
                },
            )
            if self.delay_per_tick > 0:
                await asyncio.sleep(self.delay_per_tick)
