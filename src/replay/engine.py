"""Deterministic replay engine for historical streams."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Iterable

import pandas as pd

from ....src.trading_system.core.events import TickEvent


@dataclass
class ReplayEngine:
    output_path: Path
    delay_per_tick: float = 0.0
    records: list[dict] = field(default_factory=list)

    async def from_frame(self, frame: pd.DataFrame, *, symbol: str, price_column: str = "Close") -> AsyncGenerator[TickEvent, None]:
        for index, row in frame.iterrows():
            price = float(row[price_column])
            timestamp = float(row.get("timestamp", index))
            tick = TickEvent(timestamp=timestamp, symbol=symbol, price=price, metadata=row.to_dict())
            self.records.append({"timestamp": timestamp, "symbol": symbol, "price": price, "metadata": row.to_dict()})
            yield tick
            if self.delay_per_tick > 0:
                await asyncio.sleep(self.delay_per_tick)

    def save(self) -> Path:
        self.output_path.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.records)
        path = self.output_path / "replay.parquet"
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            path = self.output_path / "replay.csv"
            frame.to_csv(path, index=False)
        return path
