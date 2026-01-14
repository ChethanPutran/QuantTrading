import yfinance as yf
import pandas_ta as ta
import pandas as pd
import pytz
from datetime import datetime, timedelta
import os
import time
import re
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from thefuzz import process
import torch
from torch.utils.data import Dataset, DataLoader
from serpapi import GoogleSearch
import joblib
import glob


# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import pandas as pd

# url="https://groww.in/stocks/intraday"
# # Setup Chrome Options
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run browser invisibly
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

# # Setup driver
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# # Open the page
# driver.get(url)
# time.sleep(5)  # Wait for page to fully load (increase if internet is slow)

# # Extract stock names and links
# stocks = driver.find_elements(By.TAG_NAME, "table")
# stock_data = []

load_dotenv(".env")

def get_todays_nifty_data():
    return load_stock_data("^NSEI",refresh=True)

def get_last_n_min_data(ticker, minutes=60, interval='1m'):
    # Download 1-minute interval data for today
    stock_data = yf.Ticker(ticker)
    data = stock_data.history(period="1d", interval=interval)

    # Get timezone-aware current time in Asia/Kolkata
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    cutoff = now - timedelta(minutes=minutes)

    # Filter data based on cutoff
    data = data[data.index >= cutoff]

    # Filter the data to last 50 minutes
    data = data[data.index >= cutoff]

    return data


def load_stock_data(ticker, interval='1m', period='1d', start=None, end=None, drop_ticker=True, save=False, file_path='yfinance_data.pkl', refresh=False):
    """
    Load stock OHLCV data using yFinance with custom start and end dates.
    :param ticker: Stock symbol (e.g., 'TCS.NS')
    :param interval: '1m', '5m', '15m', '1d', etc.
    :param period: '1d', '5d', '1mo', '3mo', '1y', etc.
    :param start: Start date (e.g., '2021-01-01')
    :param end: End date (e.g., '2021-12-31')
    :param drop_ticker: Whether to drop the ticker level if data has MultiIndex columns.
    :return: Tuple (success flag, Pandas DataFrame with OHLCV data)
    """

    try:
        if (not refresh) and os.path.exists(file_path):
            return True, pd.read_pickle(file_path)
        # If start and end dates are provided, use them, otherwise fallback to period
        if start and end:
            data = yf.download(tickers=ticker, interval=interval,
                               start=start, end=end, auto_adjust=True)
        else:
            data = yf.download(tickers=ticker, interval=interval,
                               period=period, auto_adjust=True)

        if data.empty:
            print(
                f"❌ Downloaded empty data for {ticker}. This might be due to:")
            print("- Invalid ticker symbol")
            print("- Market closed / no recent data")
            print("- Interval not supported for this period")
            return False, pd.DataFrame([])

        else:
            print(f"✅ Successfully downloaded data for {ticker}")
            # Convert the timezone to 'Asia/Kolkata'
            data.index = data.index.tz_convert('Asia/Kolkata')
            # if str(data.index.dtype) == 'datetime64[ns, UTC]':
            #     print("Hii")
            #     data.index = data.index.tz_localize("UTC").tz_convert('Asia/Kolkata')
            
            # If drop_ticker is True and data has MultiIndex, drop the second level
            if drop_ticker:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
            if save:
                data.to_pickle(file_path)
            return True, data

    except Exception as e:
        print(f"❌ Failed to download data for {ticker} due to exception:")
        print(f"   {type(e).__name__}: {e}")
    return False, pd.DataFrame([])


class DataGenerator:
    def __init__(self, tickers=[], total_days=28, interval='1m', max_days_per_request=7):
        self.total_days = total_days
        self.tickers = tickers
        self.interval = interval
        self.max_days_per_request = max_days_per_request

    def generate(self, wait_time=3):
        file_names = []
        for ticker in self.tickers:
            print(f"Generating data for {ticker}")
            data_list = []
            end_date = datetime.now()
            total_days = self.total_days
            while total_days > 0:
                days_to_fetch = min(total_days, self.max_days_per_request)
                start_date = end_date - timedelta(days=days_to_fetch)

                print(f"Fetching from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime(
                    "%Y-%m-%d")}")

                status, data = load_stock_data(ticker, start=start_date.strftime("%Y-%m-%d"),
                                               end=end_date.strftime(
                    "%Y-%m-%d"),
                    interval=self.interval, refresh=True)

                if not data.empty:
                    # data_list.append(data)
                    data_list.insert(0, data)

                total_days -= days_to_fetch
                end_date = start_date-timedelta(days=1)
                time.sleep(wait_time)

            combined_df = pd.concat(data_list)
            # remove duplicates if needed
            combined_df = combined_df[~combined_df.index.duplicated()]
            combined_df.to_csv(f"data/{ticker}_data.csv")
            file_names.append(f"data/{ticker}_data.csv")

        return file_names


def get_intraday_companynames(html, save=True):
    from bs4 import BeautifulSoup
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Find all <a> tags that have the company name
    company_tags = soup.find_all(
        "a", class_="contentPrimary swlc46ItemLink bodyBase")

    # Extract company names
    company_names = [tag.text.strip() for tag in company_tags]

    # Print the result
    res = ""
    for name in company_names:
        res += f'"{name}",\n'

    if save:
        with open("intraday_companies.txt", 'w') as f:
            f.write(res)

    return company_names


class SearchTool:
    COMPANY_COL = "NAME OF COMPANY"
    SYMBOL_COL = "SYMBOL"
    FILE_NAME = "tickers.csv"
    FAISS_INDEX_FILE_NAME = "ticker_faiss.index"
    TICKES_DATA_FILE_NAME = "ticker_data.pkl"
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    # Optional if you want LLM interface
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self):
        self.llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

    def get_matching_tickers(self, company_name: str, stock_exchange: str = "NSE", csv_path: str = "tickers.csv"):
        tickers = pd.read_csv(csv_path)
        suffix = ".NS" if stock_exchange.upper() == "NSE" else ".BO"

        # Escape special regex characters to avoid unintended behavior
        escaped_name = re.escape(company_name)

        matched = tickers[tickers[self.COMPANY_COL].str.contains(
            escaped_name, case=False, na=False)].copy()
        matched['ticker'] = matched[self.SYMBOL_COL] + suffix

        return matched[['ticker', self.COMPANY_COL]].reset_index(drop=True)

    @staticmethod
    def create_features():
        # Load your ticker CSV file
        data = pd.read_csv(SearchTool.FILE_NAME)
        tickers_df = data[SearchTool.SYMBOL_COL]

        # Create embeddings
        company_names = data[SearchTool.COMPANY_COL].fillna("").tolist()
        model = SentenceTransformer(SearchTool.MODEL_NAME)
        company_embeddings = model.encode(company_names, convert_to_numpy=True)

        # Create FAISS index
        embedding_dim = company_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(company_embeddings)

        # Save index and data
        faiss.write_index(index, SearchTool.FAISS_INDEX_FILE_NAME)
        tickers_df.to_pickle(SearchTool.TICKES_DATA_FILE_NAME)

    def search_local_ticker(self, company_query, top_k=3):
        # Load resources
        index = faiss.read_index(self.FAISS_INDEX_FILE_NAME)
        tickers_df = pd.read_pickle(self.TICKES_DATA_FILE_NAME)
        model = SentenceTransformer(self.MODEL_NAME)

        embedding = model.encode([company_query])
        D, I = index.search(np.array(embedding).astype('float32'), top_k)
        return tickers_df.iloc[I[0, 0]] + ".NS"

    def search_web_ticker(self, company_name):
        params = {
            "engine": "google",
            "q": f"{company_name} stock ticker",
            "api_key": self.SERPAPI_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        for result in results.get("organic_results", []):
            combined = result.get("title", "") + " " + \
                result.get("snippet", "")
            match = re.search(r"(NSE|BSE):\s*([A-Z0-9]+)", combined)
            if match:
                exchange = match.group(1)  # NSE or BSE
                ticker = match.group(2)    # Ticker symbol
                return ticker
            else:
                match = re.search(
                    r"\b([A-Z]{2,6}\.NS|[A-Z]{2,6}\.BO|NSE:\s?[A-Z]{2,6}|BSE:\s?[0-9]{1,6})\b", combined)
                if match:
                    return match.group(0).replace("NSE:", "").replace("BSE:", "").strip()
        return None

    def search_ticker(self, company_query):
        result = self.search_local_ticker(company_query)
        if result is not None:
            return result
        else:
            return self.search_web_ticker(company_query)

    # (Optional) Use LLM to interpret user's natural query
    def extract_company_from_user_query(self, company_name, companies):
        prompt = f"Which of the following best matches the given company?\
                    Given company :'{company_name}'\
                    List of companies :[{companies}]"
        response = self.llm.invoke(input=prompt)
        # print("Response from the llm :",response.content)
        match = re.search(r"'([A-Z]+\.(NS|BO))'", response.content)
        if match:
            ticker = match.group(1)
            # print("Extracted Ticker:", ticker)
            return ticker
        else:
            # print("Ticker not found in response.")
            return None

    # def search_local_ticker_(self,company_name, stock_exchange="NSE", score_threshold=75):
    #     suffix = ".NS" if stock_exchange.upper() == "NSE" else ".BO"
    #     names = tickers[self.COMPANY_COL].dropna().tolist()
    #     best_match, score = process.extractOne(company_name, names)

    #     if score >= score_threshold:
    #         row = tickers[tickers[self.COMPANY_COL] == best_match].iloc[0]
    #         return {
    #             "source": "local",
    #             "ticker": row[self.SYMBOL_COL] + suffix,
    #             "company": best_match
    #         }
    #     return


def get_company_tickers(html):
    search_tool = SearchTool()

    companies = get_intraday_companynames(html)
    company_tickers = []
    for company in companies:
        # Direct search
        company_details = search_tool.get_matching_tickers(company)
        if company_details.shape[0] > 1:
            # print(company)
            ticker = search_tool.extract_company_from_user_query(
                company, company_details['ticker'].to_list())
        elif company_details.shape[0] == 1:
            ticker = company_details.iloc[0].ticker
        else:
            # FAISS search
            # print(company)
            ticker = search_tool.search_local_ticker(company, top_k=3)
            # print(res)
            # ticker = search_tool.search_web_ticker(company_name=company)
        company_tickers.append(ticker)
    return company_tickers


# def preprocess_data(data, input_len, output_len, output_data=None, output_col="Output",):

#     # Generate time series seq
#     if output_data is not None:
#         n_examples, n_features = data.shape
#         n_samples = n_examples - input_len - output_len
#         X = np.zeros((n_samples, input_len, n_features))
#         y = np.zeros((n_samples, output_len))

#         for i in range(n_samples):
#             X[i] = data[i:i+input_len, :]
#             y[i] = output_data[i+input_len]

#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32)
#         return TensorDataset(X_tensor, y_tensor)

#     # Get the last input_len items
#     X = data[-input_len:]
#     return torch.tensor(X, dtype=torch.float32)


# def create_test_dataset(n_examples, n_features, input_seq_length=50, output_seq_length=1):
#     import matplotlib.pyplot as plt
#     # Simulate some data for time series (you can replace this with your actual dataset)
#     data = np.sin(np.linspace(0, 100, 1000))  # Example: sine wave
#     data = data.reshape(-1, 1)
#     plt.plot(data)
#     plt.title("data pattern")
#     plt.show()

#     data_scaled = scaler.fit_transform(data)
#     plt.plot(data_scaled)
#     plt.title("Scales data")
#     plt.show()

#     X, y = create_sequences(data_scaled, input_seq_length)
#     n_batches, *_ = X.shape

#     print("No. of examples :", n_examples)
#     print("No. of features :", n_features)
#     print("No. of batches :", n_batches)
#     print("Input sequence length :", input_seq_length)
#     print("Output sequence length :", output_seq_length)

#     # Convert data to PyTorch tensors
#     X_tensor = torch.tensor(
#         X.reshape(n_batches, input_seq_length, n_features), dtype=torch.float32)
#     y_tensor = torch.tensor(
#         y.reshape(n_batches, output_seq_length), dtype=torch.float32)

#     # Split into training and test sets
#     train_size = int(len(X_tensor) * 0.8)
#     X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
#     y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
#     y_test = y_test.numpy()

#     return X_train, y_train, X_test, y_test


# def plot_candlesticks(data):
#     fig = go.Figure(data=[
#         go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
#                        low=data['Low'], close=data['Close'], name="Candles")
#     ])
#     fig.update_layout(title="Prive vs Time",
#                       xaxis_title="Time", yaxis_title="Price")
#     fig.show()


class DatasetGenerator:
    def __init__(self):
        self.data_params = {}
        self.create_dataset()
        self.items = self.data_params.keys()

    def create_dataset(self):
        ticker_data_files = glob.glob("data")
        for ticker_file in ticker_data_files:
            data = {
                "data": None,
                "scaler": {"path": None, "load": False, "fit": False}
            }
            data['data'] = ticker_file
            ticker= ticker_file.replace(".NS_data.csv", "")
            scaler_path = "scalers/" + ticker + "_scaler.pkl"
            data['scaler']['path'] = scaler_path

            if self.load_scaler and os.path.exists(scaler_path):
                data["scaler"]["load"] = True
            else:
                data["scaler"]["fit"] = True

            self.data_params[ticker] = data

    def get_dataset(self,ticker):
        return StockDataset(self.data_params[ticker])



def scrape_intraday_stocks(url="https://groww.in/stocks/intraday"):
    # Setup Chrome Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run browser invisibly
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Setup driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    # Open the page
    driver.get(url)
    time.sleep(5)  # Wait for page to fully load (increase if internet is slow)

    # Extract stock names and links
    stocks = driver.find_elements(By.TAG_NAME, "table")
    stock_data = []
    
    for stock in stocks:
        try:
            name = stock.text
            link_element = stock.find_element(By.TAG_NAME, "a")
            link = link_element.get_attribute('href')
            stock_data.append({"Stock": name, "Link": link})
        except:
            continue

    driver.quit()
    
    # Save to dataframe
    df = pd.DataFrame(stock_data)
    return df

# Run
intraday_stocks_df = scrape_intraday_stocks()
print(intraday_stocks_df)

# Save if you want
intraday_stocks_df.to_csv('intraday_stocks_groww.csv', index=False)
print("✅ Data saved to 'intraday_stocks_groww.csv'")

    
class StockDataset(Dataset):
    def __init__(self, params, input_seq_length=50, output_seq_length=1, load_scaler=False, fit_scaler=True, scaler_path="scaler.pkl"):
        self.params = params
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.scaler_path = scaler_path
        self.fit_scaler = fit_scaler
        self.load_scaler = load_scaler
        self.train_dataset = None
        self.test_dataset = None
        self.load_data()
        

    def __len__(self):
        return len(self.train_dataset[0])

    def __getitem__(self, idx):
        return  self.train_dataset[0][idx],self.train_dataset[1][idx]

    def load_data(self, display_summary=False):
        params = self.params
        if params['scaler']['load']:
            scaler = joblib.load(params['scaler']['path'])
            print("✅ Scaler loaded.")
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))

        data_path = params['data']['path']

        df = pd.read_csv(data_path, parse_dates=['Date'])
        df.sort_values('Date', inplace=True)

        if params['scaler']['fit']:
            data_scaled = scaler.fit_transform(df)
            # Save after fitting
            joblib.dump(self.scaler, params['scaler']['path'])
            print("✅ Scaler fitted and saved.")
        else:
            data_scaled = scaler.transform(df)

        n_examples, n_features = df.shape
        X, y = self.create_sequences(
            data_scaled, output_idx=0, seq_length=self.input_seq_length)
        n_batches, *_ = X.shape
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(
            X.reshape(n_batches, self.input_seq_length, n_features), dtype=torch.float32)
        y_tensor = torch.tensor(
            y.reshape(n_batches, self.output_seq_length), dtype=torch.float32)

        # Split into training and test sets
        train_size = int(len(X_tensor) * 0.8)
        X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
        y_test = y_test.numpy()

        if display_summary:
            print(
                f"No. of examples in ticker {params['data']['ticker']}: {n_examples}")
            print(f"No. of features : {n_features}")
            print(f"No. of batches : {n_batches}")
            print(f"Input sequence length : {self.input_seq_length}")
            print(f"Output sequence length : {self.output_seq_length}")

        self.train_dataset = [X_train, y_train]
        self.test_dataset = [X_test, y_test]

    # Function to create sequences
    def create_sequences(self, data, output_idx=None, seq_length=50):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            if output_idx is not None:
                y.append(data[i+seq_length, output_idx])
            else:
                y.append(data[i+seq_length])
        return np.array(X), np.array(y)

class StockDataLoader:
    pass

if __name__ == "__main__":
    # df = load_stock_data("RELIANCE.NS")
    # print(df.head())

    html = """
<div>
    <div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/mahindra-mahindra-ltd">Mahindra &amp;
                Mahindra</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,928.80</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/reliance-industries-ltd">Reliance
                Industries</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,405.00</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/adani-enterprises-ltd">Adani
                Enterprises</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,301.30</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/britannia-industries-ltd">Britannia
                Industries</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹5,438.90</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/adani-wilmar-ltd">AWL Agri Business</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹267.05</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/marico-ltd">Marico</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹710.45</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/railtel-corporation-of-india-ltd">Railtel
                Corp</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹296.05</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/tech-mahindra-ltd">Tech Mahindra</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,503.00</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/bandhan-bank-ltd">Bandhan Bank</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹165.62</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/zomato-ltd">Eternal (Zomato)</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹232.52</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/global-vectra-helicorp-ltd">Global
                Vectra</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹220.90</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/lodha-developers-ltd">Macrotech Devs</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,329.30</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/sbi-life-insurance-company-ltd">SBI Life
                Insurance</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,765.80</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/adani-power-ltd">Adani Power</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹532.05</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/ti-financial-holdings-ltd">Cholamandalam
                Fin</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,864.10</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/bajaj-finserv-ltd">Bajaj Finserv</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,951.60</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/power-mech-projects-ltd">Power Mech
                Projects</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,619.40</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/anveshan-heavy-engineering-ltd">Anup
                Engineering</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,989.40</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/manorama-industries-ltd">Manorama Inds</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,274.70</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/hdfc-standard-life-insurance-co-ltd">HDFC
                Life Insurance</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹743.70</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/rr-kabel-ltd">RR Kabel</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,044.70</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/sangam-renewables-ltd">Waaree
                Renewables</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹959.75</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/exide-industries-ltd">Exide Industries</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹351.75</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/mankind-pharma-ltd">Mankind Pharma</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,465.30</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/waaree-energies-ltd">Waaree Energies</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,604.50</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase"
                href="/stocks/colgatepalmolive-india-ltd">Colgate-Palmolive</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,587.40</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/go-fashion-india-ltd">Go Fashion
                (India)</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹785.90</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/pokarna-ltd">Pokarna</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹903.70</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/schaeffler-india-ltd">Schaeffler India</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹3,472.90</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/ceat-ltd">CEAT</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹3,332.00</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/sonata-software-ltd">Sonata Software</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹421.55</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/tube-investments-of-india-ltd">Tube
                Investments</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,898.60</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/bharti-hexacom-ltd">Bharti Hexacom</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,689.10</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/vishal-mega-mart-ltd">Vishal Mega Mart</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹118.48</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/hindustan-aeronautics-ltd">HAL</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹4,487.90</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/bharat-electronics-ltd">Bharat
                Electronics</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹314.10</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/mazagon-dock-shipbuilders-ltd">Mazagon Dock
                Ship</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹3,057.60</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase"
                href="/stocks/garden-reach-shipbuilders-engineers-ltd">Garden Reach</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,917.00</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/cochin-shipyard-ltd">Cochin Shipyard</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,591.40</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase"
                href="/stocks/paras-defence-and-space-technologies-ltd">Paras Defence</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹1,359.55</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
        <div class="pos-rel valign-wrapper vspace-between swlc46ItemContainer"><a
                class="contentPrimary swlc46ItemLink bodyBase" href="/stocks/data-patterns-india-ltd">Data Patterns
                (I)</a>
            <div class="right-align">
                <div class="contentPrimary bodyBaseHeavy">₹2,486.60</div>
                <div class=" bodySmallHeavy contentPrimary">0.00&nbsp;(0.00%)</div>
            </div>
        </div>
    </div>
</div>
"""
    companies = get_intraday_companynames(html)
    print(companies)


    # dg = DataGenerator(tickers=tickers)
    # dg.generate(wait_time=5)

    # from torch.utils.data import DataLoader

    # batch_size = 32
    # shuffle = False

    # dataset_generator = DatasetGenerator()
    # dataset = dataset_generator.get_dataset(dataset_generator.items[0])
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)