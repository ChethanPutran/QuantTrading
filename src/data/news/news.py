from collections.abc import Iterable
import os
import re
from urllib.parse import urljoin

import pandas as pd

from ..base import BaseNewsProvider
from utils.time_utils import normalize_to_ist


HEADERS = {"User-Agent": "Mozilla/5.0"}


def _requests_get(url: str, **kwargs):
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for web news providers") from exc

    return requests.get(url, headers=HEADERS, **kwargs)


def _soup(url: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("beautifulsoup4 is required for web news providers") from exc

    return BeautifulSoup(_requests_get(url).content, "html.parser")


class NewsAPIProvider(BaseNewsProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")

    def fetch(self, query: str, count: int = 100, **kwargs) -> pd.DataFrame:
        try:
            from newsapi import NewsApiClient
        except ImportError as exc:
            raise ImportError("newsapi-python is required for NewsAPI") from exc

        api = NewsApiClient(api_key=self.api_key)
        response = api.get_everything(q=query, language="en", sort_by="relevancy")
        rows = [
            {
                "Title": article.get("title"),
                "Summary": article.get("description"),
                "Time": normalize_to_ist(article.get("publishedAt")),
                "Link": article.get("url"),
                "Source": "newsapi",
            }
            for article in response.get("articles", [])[:count]
        ]
        return pd.DataFrame(rows)


class AlphaVantageNewsProvider(BaseNewsProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ALPHAADVANTAGE_API_KEY")

    def fetch(self, query: str, **kwargs) -> pd.DataFrame:
        response = _requests_get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "symbol": query,
                "apikey": self.api_key,
            },
        )
        if response.status_code != 200:
            return pd.DataFrame()

        rows = []
        for article in response.json().get("feed", []):
            rows.append(
                {
                    "Title": article.get("title"),
                    "Time": normalize_to_ist(article.get("time_published")),
                    "Summary": article.get("summary"),
                    "SentimentScore": article.get("overall_sentiment_score"),
                    "SentimentLabel": article.get("overall_sentiment_label"),
                    "Link": article.get("url"),
                    "Source": "alphavantage",
                }
            )
        return pd.DataFrame(rows)


class GoogleNewsRSSProvider(BaseNewsProvider):
    def fetch(self, query: str = "stock market", count: int = 5, **kwargs) -> pd.DataFrame:
        try:
            import feedparser
        except ImportError as exc:
            raise ImportError("feedparser is required for Google News RSS") from exc

        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed = feedparser.parse(rss_url)
        rows = [
            {
                "Title": entry.title,
                "Summary": "",
                "Link": entry.link,
                "Time": normalize_to_ist(getattr(entry, "published", None)),
                "Source": "googlenews",
            }
            for entry in feed.entries[:count]
        ]
        return pd.DataFrame(rows)


class StaticSiteNewsProvider(BaseNewsProvider):
    def __init__(self, source: str, url: str, parser) -> None:
        self.source = source
        self.url = url
        self.parser = parser

    def fetch(self, query: str = "", count: int = 5, **kwargs) -> pd.DataFrame:
        return self.parser(self.url, count=count)


def fetch_economic_times(
    url: str = "https://economictimes.indiatimes.com/markets",
    count: int = 5,
) -> pd.DataFrame:
    soup = _soup(url)
    rows = []
    for article in soup.select("ul#just_in li")[:count]:
        link_tag = article.find("a")
        time_tag = article.find("time")
        if not link_tag:
            continue
        rows.append(
            {
                "Title": link_tag.get("title"),
                "Summary": "",
                "Time": normalize_to_ist(time_tag.get("data-time") if time_tag else None),
                "Link": urljoin(url, link_tag.get("href", "")),
                "Source": "economictimes",
            }
        )
    return pd.DataFrame(rows)


def fetch_pulse_zerodha(
    url: str = "https://pulse.zerodha.com/",
    count: int = 5,
) -> pd.DataFrame:
    soup = _soup(url)
    rows = []
    for article in soup.select("ul#news li.box.item")[:count]:
        link = article.select_one("a")
        body = article.select_one("div.desc")
        date = article.select_one("span.date")
        if not link:
            continue
        rows.append(
            {
                "Title": link.get_text(strip=True),
                "Summary": body.get_text(strip=True) if body else "",
                "Link": link.get("href"),
                "Time": normalize_to_ist(date.get("title") if date else None),
                "Source": "pulsezerodha",
            }
        )
    return pd.DataFrame(rows)


def fetch_moneycontrol(
    url: str = "https://www.moneycontrol.com/news/business/markets/",
    count: int = 5,
) -> pd.DataFrame:
    soup = _soup(url)
    rows = []
    pattern = r"\b([A-Za-z]{3,9} \d{2}, \d{4} \d{2}:\d{2} [APap][Mm] [A-Za-z]{2,4})\b"
    matches = re.findall(pattern, str(soup))

    for article in soup.select("li.clearfix")[:count]:
        link = article.find("a")
        if not link:
            continue
        body = article.find("p")
        rows.append(
            {
                "Title": link.get("title", "").strip(),
                "Summary": body.get_text(strip=True) if body else "",
                "Time": normalize_to_ist(matches[0] if matches else None),
                "Link": link.get("href"),
                "Source": "moneycontrol",
            }
        )
    return pd.DataFrame(rows)


def fetch_all_market_news(
    providers: Iterable[BaseNewsProvider] | None = None,
    unique: bool = True,
    latest: bool = True,
) -> pd.DataFrame:
    providers = providers or [
        GoogleNewsRSSProvider(),
        StaticSiteNewsProvider("economictimes", "https://economictimes.indiatimes.com/markets", fetch_economic_times),
        StaticSiteNewsProvider("moneycontrol", "https://www.moneycontrol.com/news/business/markets/", fetch_moneycontrol),
        StaticSiteNewsProvider("pulsezerodha", "https://pulse.zerodha.com/", fetch_pulse_zerodha),
    ]
    frames = [provider.fetch("stock market") for provider in providers]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    if latest and "Time" in output:
        output = output.sort_values(by="Time", ascending=False)
    if unique:
        subset = "Link" if "Link" in output else None
        output = output.drop_duplicates(subset=subset)
    return output.reset_index(drop=True)


class NewsExtractor:
    """Backward-compatible facade over focused news providers."""

    def get_market_news_newsapi(self, company_name, count=100):
        return NewsAPIProvider().fetch(company_name, count=count)

    def get_market_news_alphaad(self, ticker):
        return AlphaVantageNewsProvider().fetch(ticker)

    def get_market_news_economictimes(self, url="https://economictimes.indiatimes.com/markets", count=5):
        return fetch_economic_times(url, count=count)

    def get_market_news_pulsezerodha(self, url="https://pulse.zerodha.com/", count=5):
        return fetch_pulse_zerodha(url, count=count)

    def get_market_news_moneycontrol(self, url="https://www.moneycontrol.com/news/business/markets/", count=5):
        return fetch_moneycontrol(url, count=count)

    def get_market_news_googlenews(self, count=5):
        return GoogleNewsRSSProvider().fetch("stock market", count=count)

    def get_market_news_all(self, display=False, unique=True, latest=True):
        return fetch_all_market_news(unique=unique, latest=latest)

    def fetch_company_news(self, company_name, max_per_site=30):
        query = f'"{company_name}" stock market'
        return GoogleNewsRSSProvider().fetch(query, count=max_per_site)
