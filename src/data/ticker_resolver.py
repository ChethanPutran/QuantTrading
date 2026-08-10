import os
import re

import numpy as np
import pandas as pd

from .base import BaseTickerResolver


class CsvTickerResolver(BaseTickerResolver):
    COMPANY_COL = "NAME OF COMPANY"
    SYMBOL_COL = "SYMBOL"

    def __init__(self, csv_path: str = "tickers.csv") -> None:
        self.csv_path = csv_path

    def resolve(
        self,
        company_name: str,
        stock_exchange: str = "NSE",
        **kwargs,
    ) -> str | None:
        matches = self.get_matching_tickers(company_name, stock_exchange)
        if matches.empty:
            return None
        return str(matches.iloc[0]["ticker"])

    def get_matching_tickers(
        self,
        company_name: str,
        stock_exchange: str = "NSE",
    ) -> pd.DataFrame:
        tickers = pd.read_csv(self.csv_path)
        suffix = ".NS" if stock_exchange.upper() == "NSE" else ".BO"
        escaped_name = re.escape(company_name)
        matched = tickers[
            tickers[self.COMPANY_COL].str.contains(escaped_name, case=False, na=False)
        ].copy()
        matched["ticker"] = matched[self.SYMBOL_COL] + suffix
        return matched[["ticker", self.COMPANY_COL]].reset_index(drop=True)


class FaissTickerResolver(BaseTickerResolver):
    COMPANY_COL = "NAME OF COMPANY"
    SYMBOL_COL = "SYMBOL"
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        tickers_csv: str = "tickers.csv",
        index_path: str = "ticker_faiss.index",
        data_path: str = "ticker_data.pkl",
    ) -> None:
        self.tickers_csv = tickers_csv
        self.index_path = index_path
        self.data_path = data_path

    def create_index(self) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "faiss and sentence-transformers are required for semantic ticker search"
            ) from exc

        data = pd.read_csv(self.tickers_csv)
        company_names = data[self.COMPANY_COL].fillna("").tolist()
        model = SentenceTransformer(self.MODEL_NAME)
        embeddings = model.encode(company_names, convert_to_numpy=True)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        data[self.SYMBOL_COL].to_pickle(self.data_path)

    def resolve(self, company_name: str, top_k: int = 3, **kwargs) -> str | None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "faiss and sentence-transformers are required for semantic ticker search"
            ) from exc

        if not os.path.exists(self.index_path) or not os.path.exists(self.data_path):
            return None

        index = faiss.read_index(self.index_path)
        tickers_df = pd.read_pickle(self.data_path)
        model = SentenceTransformer(self.MODEL_NAME)
        embedding = model.encode([company_name])
        _, indexes = index.search(np.asarray(embedding).astype("float32"), top_k)
        return str(tickers_df.iloc[indexes[0, 0]]) + ".NS"


class WebTickerResolver(BaseTickerResolver):
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("SERPAPI_KEY")

    def resolve(self, company_name: str, **kwargs) -> str | None:
        try:
            from serpapi import GoogleSearch
        except ImportError as exc:
            raise ImportError("serpapi is required for web ticker search") from exc

        search = GoogleSearch(
            {
                "engine": "google",
                "q": f"{company_name} stock ticker",
                "api_key": self.api_key,
            }
        )
        results = search.get_dict()
        for result in results.get("organic_results", []):
            combined = f"{result.get('title', '')} {result.get('snippet', '')}"
            match = re.search(r"(NSE|BSE):\s*([A-Z0-9]+)", combined)
            if match:
                return match.group(2)
            match = re.search(
                r"\b([A-Z]{2,6}\.NS|[A-Z]{2,6}\.BO|NSE:\s?[A-Z]{2,6}|BSE:\s?[0-9]{1,6})\b",
                combined,
            )
            if match:
                return match.group(0).replace("NSE:", "").replace("BSE:", "").strip()
        return None


class SearchTool:
    """Compatibility facade for old data_fetcher.SearchTool."""

    COMPANY_COL = CsvTickerResolver.COMPANY_COL
    SYMBOL_COL = CsvTickerResolver.SYMBOL_COL
    FILE_NAME = "tickers.csv"
    FAISS_INDEX_FILE_NAME = "ticker_faiss.index"
    TICKES_DATA_FILE_NAME = "ticker_data.pkl"

    def get_matching_tickers(
        self,
        company_name: str,
        stock_exchange: str = "NSE",
        csv_path: str = "tickers.csv",
    ) -> pd.DataFrame:
        return CsvTickerResolver(csv_path).get_matching_tickers(
            company_name,
            stock_exchange,
        )

    @staticmethod
    def create_features():
        FaissTickerResolver(
            tickers_csv=SearchTool.FILE_NAME,
            index_path=SearchTool.FAISS_INDEX_FILE_NAME,
            data_path=SearchTool.TICKES_DATA_FILE_NAME,
        ).create_index()

    def search_local_ticker(self, company_query, top_k=3):
        return FaissTickerResolver(
            index_path=self.FAISS_INDEX_FILE_NAME,
            data_path=self.TICKES_DATA_FILE_NAME,
        ).resolve(company_query, top_k=top_k)

    def search_web_ticker(self, company_name):
        return WebTickerResolver().resolve(company_name)

    def search_ticker(self, company_query):
        return self.search_local_ticker(company_query) or self.search_web_ticker(
            company_query
        )

    def extract_company_from_user_query(self, company_name, companies):
        for company in companies:
            if company_name.lower() in str(company).lower():
                return company
        return companies[0] if companies else None


def get_intraday_companynames(html: str, save: bool = False) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("beautifulsoup4 is required to parse company names") from exc

    soup = BeautifulSoup(html, "html.parser")
    company_tags = soup.find_all(
        "a",
        class_="contentPrimary swlc46ItemLink bodyBase",
    )
    company_names = [tag.text.strip() for tag in company_tags]

    if save:
        with open("intraday_companies.txt", "w", encoding="utf-8") as file:
            file.write("".join(f'"{name}",\n' for name in company_names))

    return company_names


def get_company_tickers(html: str) -> list[str | None]:
    search_tool = SearchTool()
    tickers = []

    for company in get_intraday_companynames(html):
        matches = search_tool.get_matching_tickers(company)
        if len(matches) > 1:
            ticker = search_tool.extract_company_from_user_query(
                company,
                matches["ticker"].to_list(),
            )
        elif len(matches) == 1:
            ticker = matches.iloc[0].ticker
        else:
            ticker = search_tool.search_local_ticker(company, top_k=3)
        tickers.append(ticker)

    return tickers
