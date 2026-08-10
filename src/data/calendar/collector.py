import asyncio
import aiohttp

from .schema import CalendarEvent
from ..base import BaseDataCollector

class CalendarCollector(BaseDataCollector):

    def __init__(self, url: str):
        self.url = url

    async def run(self):

        async with aiohttp.ClientSession() as session:

            while True:
                async with session.get(self.url) as response:

                    events = await response.json()

                    for item in events:
                        event = CalendarEvent(
                            timestamp=item["timestamp"],
                            event_name=item["event"],
                            country=item["country"],
                            impact=item["impact"],
                            category=item["category"],
                            expected_volatility=item.get("expected_vol", 0.0)
                        )

                        print(event)

                await asyncio.sleep(3600)