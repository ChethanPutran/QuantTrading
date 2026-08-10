from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


FeatureDict = dict[str, float]


class BaseFeatureProvider(ABC):
    """Fetches raw source data for a feature group."""

    @abstractmethod
    def fetch(self, ticker: str) -> Mapping[str, Any] | pd.DataFrame:
        pass


class BaseFeatureTransformer(ABC):
    """Transforms source data into named numeric features."""

    @abstractmethod
    def transform(self, data: Any) -> FeatureDict | pd.DataFrame:
        pass


class BaseFeatureBuilder(ABC):
    """Builds a merged feature dictionary from multiple sources."""

    @abstractmethod
    def build(self, ticker: str) -> FeatureDict:
        pass


class BaseFeaturePipeline(ABC):
    """Converts raw feature data into model-ready arrays."""

    @abstractmethod
    def transform(self, data: dict) -> np.ndarray:
        pass


class BaseFilter(ABC):
    """Noise filtering interface."""

    @abstractmethod
    def update(self, observation: float) -> np.ndarray:
        pass
