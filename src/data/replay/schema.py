
from attr import dataclass



@dataclass
class ReplayEvent:
    timestamp: int

    event_type: str

    payload: dict