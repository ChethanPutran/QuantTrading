from abc import ABC, abstractmethod


class BaseConstraints(ABC):
    """Abstract constraint interface."""

    @abstractmethod
    def validate(self, action: float, state: dict) -> bool:
        pass


class ConstraintManager(BaseConstraints):
    """Applies all constraints."""

    def __init__(self, constraints: dict):
        self.constraints = constraints

    def validate(self, action: float, state: dict) -> bool:
        pos_limits = self.constraints["position"]

        if action > pos_limits["max_position"]:
            return False
        if action < pos_limits["min_position"]:
            return False

        return True