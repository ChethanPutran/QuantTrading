
from dataclasses import dataclass



@dataclass
class CalendarEvent:
    timestamp: int

    event_name: str

    country: str

    impact: str

    category: str

    expected_volatility: float