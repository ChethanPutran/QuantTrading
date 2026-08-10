
from attr import dataclass


@dataclass
class MacroEvent:
    timestamp: int

    country: str

    indicator: str

    actual: float
    forecast: float
    previous: float

    surprise: float

    importance: int