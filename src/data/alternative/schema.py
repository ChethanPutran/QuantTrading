
from attr import dataclass



@dataclass
class AlternativeEvent:
    timestamp: int

    source_type: str

    metric_name: str

    value: float

    confidence: float