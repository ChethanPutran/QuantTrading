
from dataclasses import dataclass



@dataclass
class QualityEvent:
    timestamp: int

    source: str

    latency_ms: float

    missing_fields: int

    anomaly_score: float