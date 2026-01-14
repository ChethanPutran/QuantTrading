import pandas as pd
from quant_trading.src.pre_processing.data.grow_automation import PASSWORD, USERNAME
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
import requests
from bs4 import BeautifulSoup

# Start a session
session = requests.Session()

# URL where the login form is submitted
login_url = 'https://groww.in/screener/stocks/intraday'

# Your login payload (match the form field names exactly)
payload = {
    'username': USERNAME,
    'password': PASSWORD
}

# Post login data
response = session.get(login_url, data=payload)


# Configure ChromeDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Optional: run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Path to your ChromeDriver executable
# service = Service('/path/to/chromedriver')  # Change this!

# Start browser
driver = webdriver.Chrome(options=chrome_options)

# Visit NSE website to generate cookies
driver.get("https://www.nseindia.com")

# Get cookies from Selenium
selenium_cookies = driver.get_cookies()
driver.quit()

# Convert cookies to requests format
cookies_dict = {cookie['name']: cookie['value'] for cookie in selenium_cookies}


def get_nse_stock_data(symbol):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={symbol.upper()}"
    }

    # NSE blocks direct requests unless session cookies are set
    session = requests.Session()
    session.headers.update(headers)

    # Make initial request to get cookies
    session.get("https://www.nseindia.com", timeout=5)

    response = session.get(url, timeout=5)

    if response.status_code == 200:
        data = response.json()
        info = {
            "symbol": symbol.upper(),
            "lastPrice": data["priceInfo"]["lastPrice"],
            "previousClose": data["priceInfo"]["previousClose"],
            "open": data["priceInfo"]["open"],
            "dayHigh": data["priceInfo"]["intraDayHighLow"]["max"],
            "dayLow": data["priceInfo"]["intraDayHighLow"]["min"],
            "volume": data["priceInfo"]["totalTradedVolume"]
        }
        return info
    else:
        raise Exception(
            f"Failed to fetch data. Status Code: {response.status_code}")


# Example usage
ticker = "TCS"
try:
    stock_data = get_nse_stock_data(ticker)
    print(stock_data)
except Exception as e:
    print(e)


def get_most_traded_nse_stocks():
    url = "https://www.nseindia.com/api/live-analysis-most-active-securities?index=volume"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/live-equity-market"
    }

    session = requests.Session()
    session.headers.update(headers)

    # Initial request to set cookies
    res = session.get("https://www.nseindia.com", timeout=10)

    headers["Cookie"] = res.cookies.get_dict()["_abck"]
    response = session.get(url, timeout=5, headers=headers)

    print(response.status_code)
    if response.status_code == 200:
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df = df[["symbol", "lastPrice", "volume", "value", "pChange"]]
        df.columns = ["Symbol", "Last Price",
                      "Volume", "Turnover ₹", "% Change"]
        return df
    else:
        raise Exception("Failed to fetch data from NSE")


# Example usage
try:
    most_traded = get_most_traded_nse_stocks()
    print(most_traded)
except Exception as e:
    print(e)
