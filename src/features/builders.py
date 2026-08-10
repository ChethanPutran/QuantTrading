from collections.abc import Iterable

from .base import BaseFeatureBuilder, BaseFeatureProvider, BaseFeatureTransformer
from .fundamental import (
    FundamentalFeatureTransformer,
    compute_quarterly_metrics as compute_quarterly_metrics_from_statements,
)
from .macro import MacroFeatureTransformer
from .providers import (
    FredMacroProvider,
    OpenBBSentimentProvider,
    YFinanceFundamentalProvider,
    YFinancePriceProvider,
)
from .sentiment import SentimentFeatureTransformer
from .technical import latest_technical_features


class ProviderFeatureSpec:
    def __init__(
        self,
        name: str,
        provider: BaseFeatureProvider,
        transformer: BaseFeatureTransformer,
    ) -> None:
        self.name = name
        self.provider = provider
        self.transformer = transformer

    def build(self, ticker: str) -> dict[str, float]:
        raw = self.provider.fetch(ticker)
        features = self.transformer.transform(raw)
        if hasattr(features, "iloc"):
            latest = features.iloc[-1]
            return {
                str(name): float(value)
                for name, value in latest.items()
                if value == value
            }
        return dict(features)


class CallableFeatureSpec:
    def __init__(self, name: str, build_fn) -> None:
        self.name = name
        self.build_fn = build_fn

    def build(self, ticker: str) -> dict[str, float]:
        return dict(self.build_fn(ticker))


class CompositeFeatureBuilder(BaseFeatureBuilder):
    def __init__(self, specs: Iterable[ProviderFeatureSpec | CallableFeatureSpec]):
        self.specs = list(specs)

    def build(self, ticker: str) -> dict[str, float]:
        features: dict[str, float] = {}

        for spec in self.specs:
            features.update(spec.build(ticker))

        return features


class FeatureBuilder:
    """Backward-compatible facade for building ticker-level features."""

    def __init__(
        self,
        ticker: str,
        include_sentiment: bool = True,
        include_macro: bool = True,
    ) -> None:
        self.ticker = ticker
        self.include_sentiment = include_sentiment
        self.include_macro = include_macro
        self.features: dict[str, float] = {}

    def get_technical_features(self) -> dict[str, float]:
        raw = YFinancePriceProvider().fetch(self.ticker)
        return latest_technical_features(raw)

    def add_technical_features(self, features: dict[str, float]) -> None:
        self.features.update(features)

    def get_fundamental_features(self) -> dict[str, float]:
        raw = YFinanceFundamentalProvider().fetch(self.ticker)
        return FundamentalFeatureTransformer().transform(raw)

    def add_fundamental_features(self, features: dict[str, float]) -> None:
        self.features.update(features)

    def get_macroeconomic_features(self) -> dict[str, float]:
        raw = FredMacroProvider().fetch(self.ticker)
        return MacroFeatureTransformer().transform(raw)

    def add_macroeconomic_features(self, features: dict[str, float]) -> None:
        self.features.update(features)

    def get_market_sentiments_features(self) -> dict[str, float]:
        raw = OpenBBSentimentProvider().fetch(self.ticker)
        return SentimentFeatureTransformer().transform(raw)

    def add_market_sentiments_features(self, features: dict[str, float]) -> None:
        self.features.update(features)

    def build_full_feature_vector(self) -> dict[str, float]:
        self.add_fundamental_features(self.get_fundamental_features())
        self.add_technical_features(self.get_technical_features())

        if self.include_macro:
            self.add_macroeconomic_features(self.get_macroeconomic_features())

        if self.include_sentiment:
            self.add_market_sentiments_features(
                self.get_market_sentiments_features()
            )

        return self.features

    def get_features(self) -> dict[str, float]:
        return self.features


def compute_quarterly_metrics(ticker: str, start_date, end_date):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for quarterly metrics") from exc

    stock = yf.Ticker(ticker)
    return compute_quarterly_metrics_from_statements(
        income=stock.quarterly_income_stmt,
        balance=stock.quarterly_balance_sheet,
        info=stock.info,
        start_date=start_date,
        end_date=end_date,
    )


__all__ = [
    "CallableFeatureSpec",
    "CompositeFeatureBuilder",
    "FeatureBuilder",
    "ProviderFeatureSpec",
    "compute_quarterly_metrics",
]
