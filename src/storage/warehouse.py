"""Storage abstraction with Redis/DuckDB/Parquet friendly interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:  # optional dependency
    import duckdb
except Exception:  # pragma: no cover - optional dependency
    duckdb = None

try:  # optional dependency
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None


@dataclass
class AnalyticsStore:
    root: Path
    cache: dict[str, Any] = field(default_factory=dict)
    redis_url: str | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._redis_client = None
        if self.redis_url and redis is not None:
            try:
                self._redis_client = redis.from_url(self.redis_url)
            except Exception:
                self._redis_client = None

    def put_frame(self, name: str, frame: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{name}.parquet"
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            path = self.root / f"{name}.csv"
            frame.to_csv(path, index=False)
        self.cache[name] = frame
        return path

    def append_records(self, name: str, records: list[dict[str, Any]]) -> Path:
        frame = pd.DataFrame(records)
        return self.put_frame(name, frame)

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self.cache[name] = payload
        return path

    def save_model_snapshot(self, name: str, payload: dict[str, Any]) -> Path:
        return self.write_json(f"model_{name}", payload)

    def save_pattern_db(self, name: str, payload: dict[str, Any]) -> Path:
        return self.write_json(f"patterns_{name}", payload)

    def save_replay_log(self, name: str, records: list[dict[str, Any]]) -> Path:
        return self.append_records(f"replay_{name}", records)

    def save_events(self, name: str, records: list[dict[str, Any]]) -> Path:
        return self.append_records(f"events_{name}", records)

    def publish_state(self, channel: str, payload: dict[str, Any]) -> None:
        if self._redis_client is None:
            return
        try:
            self._redis_client.publish(channel, json.dumps(payload, default=str))
        except Exception:
            return

    def query(self, sql: str) -> pd.DataFrame:
        if duckdb is None:
            raise RuntimeError("duckdb is not available")
        return duckdb.query(sql).df()

class ReplayStore(AnalyticsStore):
    """Replay store for storing historical events and replay logs."""

    def save_replay_log(self, name: str, records: list[dict[str, Any]]) -> Path:
        return self.append_records(f"replay_{name}", records)

    def save_events(self, name: str, records: list[dict[str, Any]]) -> Path:
        return self.append_records(f"events_{name}", records)
    

class StateStore(AnalyticsStore):
    """State store for storing features, trades, and other stateful data."""

    def put_frame(self, name: str, frame: pd.DataFrame) -> Path:
        return super().put_frame(name, frame)