import asyncio
import aiohttp

from .schema import MacroEvent
from ..base import BaseDataCollector

class MacroCollector(BaseDataCollector):

    def __init__(self, indicators: list[str]):
        self.indicators = indicators

    async def fetch_indicator(self, session, indicator):
        url = f"https://api.example.com/macro/{indicator}"

        async with session.get(url) as response:
            data = await response.json()

            surprise = data["actual"] - data["forecast"]

            event = MacroEvent(
                timestamp=data["timestamp"],
                country=data["country"],
                indicator=indicator,
                actual=data["actual"],
                forecast=data["forecast"],
                previous=data["previous"],
                surprise=surprise,
                importance=data["importance"]
            )

            print(event)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [
                    self.fetch_indicator(session, i)
                    for i in self.indicators
                ]

                await asyncio.gather(*tasks)

                await asyncio.sleep(300)