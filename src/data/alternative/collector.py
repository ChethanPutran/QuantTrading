import asyncio

from .schema import AlternativeEvent
from ..base import BaseDataCollector

class AlternativeCollector(BaseDataCollector):

    async def fetch_social_sentiment(self):

        event = AlternativeEvent(
            timestamp=0,
            source_type="social",
            metric_name="market_sentiment",
            value=0.62,
            confidence=0.91
        )

        print(event)

    async def run(self):
        while True:
            await self.fetch_social_sentiment()
            await asyncio.sleep(60)