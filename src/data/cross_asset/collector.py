import asyncio
import aiohttp

from .schema import CrossAssetEvent
from ..base import BaseDataCollector

class CrossAssetCollector(BaseDataCollector):

    def __init__(self, symbols: list[str]):
        self.symbols = symbols

    async def fetch_symbol(self, session, symbol):
        url = f"https://api.example.com/{symbol}"

        async with session.get(url) as response:
            data = await response.json()

            event = CrossAssetEvent(
                timestamp=data["timestamp"],
                asset_class=data["asset_class"],
                symbol=symbol,
                price=data["price"],
                return_1m=data["return_1m"],
                volatility=data["volatility"],
                correlation_market=data["corr_market"]
            )

            print(event)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [
                    self.fetch_symbol(session, s)
                    for s in self.symbols
                ]

                await asyncio.gather(*tasks)

                await asyncio.sleep(5)