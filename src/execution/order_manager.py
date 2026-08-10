from abc import ABC, abstractmethod


class BaseOrderManager(ABC):
    """Handles order placement."""

    @abstractmethod
    def execute(self, action: int) -> None:
        pass