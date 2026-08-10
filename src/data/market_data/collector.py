import asyncio
import json
import websockets

from .schema import TickEvent
from ..base import BaseDataCollector

class MarketDataCollector(BaseDataCollector):
    def __init__(self, url: str, symbols: list[str]):
        self.url = url
        self.symbols = symbols

    async def run(self):
        await self.connect()

    async def connect(self):
        async with websockets.connect(self.url) as ws:

            subscribe_msg = {
                "type": "subscribe",
                "symbols": self.symbols
            }

            await ws.send(json.dumps(subscribe_msg))

            while True:
                msg = await ws.recv()
                await self.process(msg)

    async def process(self, raw_msg: str):
        data = json.loads(raw_msg)

        event = TickEvent(
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            bid=data["bid"],
            ask=data["ask"],
            last=data["last"],
            volume=data["volume"],
            spread=data["ask"] - data["bid"],
            exchange=data.get("exchange", "unknown")
        )

        print(event)