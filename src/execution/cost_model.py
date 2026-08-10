from abc import ABC, abstractmethod


class BaseCostModel(ABC):
    """Transaction cost model."""

    @abstractmethod
    def compute(self, action: int) -> float:
        pass