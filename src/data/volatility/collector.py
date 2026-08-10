import numpy as np

from .schema import VolatilityEvent
from ..base import BaseDataCollector

class VolatilityCollector(BaseDataCollector):

    def __init__(self):
        self.returns = []

    def update(self, timestamp, symbol, price):

        self.returns.append(price)

        if len(self.returns) < 50:
            return

        arr = np.diff(np.log(self.returns[-50:]))

        realized_vol = np.std(arr)

        regime = "high" if realized_vol > 0.02 else "normal"

        event = VolatilityEvent(
            timestamp=timestamp,
            symbol=symbol,
            realized_vol=realized_vol,
            implied_vol=0.0,
            vix=0.0,
            volatility_regime=regime
        )

        print(event)