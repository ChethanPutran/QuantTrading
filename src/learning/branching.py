from abc import ABC, abstractmethod


class BaseBranchingStrategy(ABC):
    """Controls pattern branching logic."""

    @abstractmethod
    def should_branch(self, context: dict) -> bool:
        pass