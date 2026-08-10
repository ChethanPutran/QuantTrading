"""Replay-oriented backtesting helper."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..replay.engine import ReplayEngine


@dataclass
class BacktestEngine:
    replay_engine: ReplayEngine

    async def run(self, frame: pd.DataFrame, symbol: str) -> None:
        async for _ in self.replay_engine.from_frame(frame, symbol=symbol):
            pass
