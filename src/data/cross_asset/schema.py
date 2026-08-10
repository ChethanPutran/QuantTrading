
from dataclasses import dataclass

class Indices:
    SPX = "SPX" # S&P 500 Index
    VIX = "VIX" # CBOE Volatility Index
    DXY = "DXY" # US Dollar Index

class Commodities:
    GOLD = "GOLD" # Gold
    WTI_OIL = "WTI_OIL" # Oil
    NATURAL_GAS = "NATURAL_GAS" # Natural Gas
    SILVER = "SILVER" # Silver


class Bonds:
    US10Y = "US10Y" # 10-year US Treasury yield
    US30Y = "US30Y" # 30-year US Treasury yield
    US2Y = "US2Y" # 2-year US Treasury yield
    US5Y = "US5Y" # 5-year US Treasury yield

class Forex:
    EURUSD = "EURUSD" # Euro/US Dollar
    USDJPY = "USDJPY" # US Dollar/Japanese Yen
    GBPUSD = "GBPUSD" # British Pound/US Dollar
    AUDUSD = "AUDUSD" # Australian Dollar/US Dollar
    USDCAD = "USDCAD" # US Dollar/Canadian Dollar
    
class AssetClass:
    EQUITY = "equity" # Stocks and ETFs
    COMMODITY = "commodity" # Gold, oil, etc.
    CURRENCY = "currency" # Forex pairs like EUR/USD, USD/JPY, etc.
    CRYPTO = "crypto" # Cryptocurrencies like Bitcoin, Ethereum, etc.
    BOND = "bond" # Government and corporate bonds
    INDEX = "index" # Broad market indices like S&P 500, Nasdaq, etc.
    GOLD = "gold" # Gold
    WTI_OIL = "wti_oil" # Oil
    US10Y = "us10y" # 10-year US Treasury yield
    DXY = "dxy" # US Dollar Index
    SPX = "spx" # S&P 500 Index
    VIX = "vix" # CBOE Volatility Index

# Capture inter-market dependencies.
@dataclass
class CrossAssetEvent:
    timestamp: int

    asset_class: AssetClass
    symbol: str

    price: float
    return_1m: float

    volatility: float

    correlation_market: float