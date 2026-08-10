import asyncio
import aiohttp

from .schema import NewsEvent
from .nlp import analyze_text
from ..base import BaseDataCollector

class NewsCollector(BaseDataCollector):

    def __init__(self, feeds: list[str]):
        self.feeds = feeds

    async def fetch_feed(self, session, url):
        async with session.get(url) as response:
            articles = await response.json()

            for article in articles:
                features = analyze_text(article["headline"])

                event = NewsEvent(
                    timestamp=article["timestamp"],
                    headline=article["headline"],
                    source=article["source"],
                    sentiment=features["sentiment"],
                    uncertainty=features["uncertainty"],
                    relevance=features["relevance"],
                    topic=features["topic"],
                    entities=features["entities"],
                    embedding=features["embedding"]
                )

                print(event)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                tasks = [
                    self.fetch_feed(session, feed)
                    for feed in self.feeds
                ]

                await asyncio.gather(*tasks)

                await asyncio.sleep(10)