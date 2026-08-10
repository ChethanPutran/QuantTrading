from collections.abc import Mapping

import numpy as np
import pandas as pd

from data.base import BaseSentimentAnalyzer


class VaderSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError("vaderSentiment is required for VADER scoring") from exc

        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> Mapping[str, float]:
        scores = self.analyzer.polarity_scores(str(text))
        return {
            "Neg": scores.get("neg", 0.0),
            "Neu": scores.get("neu", 0.0),
            "Pos": scores.get("pos", 0.0),
            "Compound": scores.get("compound", 0.0),
        }


class SentimentFeatureTransformer:
    """Simple feature transformer that converts news/article data to numeric sentiment features."""

    def __init__(self, analyzer: BaseSentimentAnalyzer | None = None) -> None:
        self.analyzer = analyzer or VaderSentimentAnalyzer()

    def transform(self, raw) -> pd.DataFrame:
        """
        Accepts either a pandas DataFrame of articles (with Title/Summary columns)
        or a list of text items. Returns a single-row DataFrame with aggregated
        sentiment features.
        """
        if raw is None:
            return pd.DataFrame([{"sentiment_compound": 0.0}])

        if isinstance(raw, pd.DataFrame):
            scored = score_news_dataframe(raw, self.analyzer)
            compound = scored['Compound'].astype(float).mean() if 'Compound' in scored else 0.0
            return pd.DataFrame([{"sentiment_compound": float(compound)}])

        # Fallback: list or single string
        if isinstance(raw, (list, tuple)):
            vals = [self.analyzer.score(str(x))['Compound'] for x in raw]
            return pd.DataFrame([{"sentiment_compound": float(np.mean(vals) if vals else 0.0)}])

        # Single text
        val = self.analyzer.score(str(raw))['Compound']
        return pd.DataFrame([{"sentiment_compound": float(val)}])


class TextBlobSentimentAnalyzer(BaseSentimentAnalyzer):
    def score(self, text: str) -> Mapping[str, float]:
        try:
            from textblob import TextBlob
        except ImportError as exc:
            raise ImportError("textblob is required for TextBlob scoring") from exc

        polarity = TextBlob(str(text)).sentiment.polarity
        return {
            "Neg": float(polarity < 0),
            "Neu": float(polarity == 0),
            "Pos": float(polarity > 0),
            "Compound": polarity,
        }


def score_news_dataframe(
    news_df: pd.DataFrame,
    analyzer: BaseSentimentAnalyzer | None = None,
    title_col: str = "Title",
    summary_col: str = "Summary",
) -> pd.DataFrame:
    analyzer = analyzer or VaderSentimentAnalyzer()
    output = news_df.copy()

    def score_row(row):
        text = f"{row.get(title_col, '')} {row.get(summary_col, '')}"
        return pd.Series(analyzer.score(text))

    scores = output.apply(score_row, axis=1)
    return pd.concat([output, scores], axis=1)


def analyze_sentiment(articles) -> tuple[int, int, int]:
    analyzer = TextBlobSentimentAnalyzer()
    positive = negative = neutral = 0

    for article in articles:
        if isinstance(article, Mapping):
            text = f"{article.get('title', '')} {article.get('description', '')}"
        else:
            text = str(article)
        score = analyzer.score(text)["Compound"]
        if score > 0:
            positive += 1
        elif score < 0:
            negative += 1
        else:
            neutral += 1

    return positive, negative, neutral


def aggregate_sentiment_scores(*scores: float) -> float:
    usable = [score for score in scores if score is not None and score == score]
    return float(np.mean(usable)) if usable else 0.0


def select_stocks_based_on_sentiment(sentiment_score: float) -> str:
    if sentiment_score > 0.2:
        return "BUY"
    if sentiment_score < -0.2:
        return "SELL"
    return "HOLD"


class StockSelector:
    def __init__(self, analyzer: BaseSentimentAnalyzer | None = None) -> None:
        self.analyzer = analyzer or VaderSentimentAnalyzer()

    def get_sentiment_scores(self, item):
        return self.analyzer.score(f"{getattr(item, 'Title', '')} {getattr(item, 'Summary', '')}")

    def add_new_sentiments(self, news_df):
        return score_news_dataframe(news_df, self.analyzer)
