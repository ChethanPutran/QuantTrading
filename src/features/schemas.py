from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MarketBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Any | None = None
    symbol: str | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True)
class FeatureSet:
    values: dict[str, float] = field(default_factory=dict)
    source: str | None = None

    def merge(self, other: "FeatureSet") -> "FeatureSet":
        return FeatureSet(values={**self.values, **other.values}, source=self.source)


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class FeatureConfig:
    feature_names: tuple[str, ...] | None = None
    include_derived: bool = True
    fill_value: float = 0.0
