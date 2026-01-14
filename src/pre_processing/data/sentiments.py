# !pip install tweepy vaderSentiment praw
# !pip install streamlit
# !pip install spacy
# !python -m spacy download en_core_web_sm
# !pip install newsapi-python
# !pip install feedparser

import requests
import pandas as pd
from dotenv import load_dotenv
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import numpy as np
import praw
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import streamlit as st
import sqlite3
import spacy
from urllib.parse import urljoin
from datetime import datetime
import feedparser
import re
from dateutil.tz import gettz, tzutc
import pytz
from dateutil import parser as date_parser
from langchain_nvidia import ChatNVIDIA
import os
import yaml
# from app.predict import run_prediction_for_all_tickers  # Import from your prediction module
# from app.database import get_prediction_history  # Import from your SQLite functions
# from app.sentiment import get_twitter_sentiment, get_reddit_sentiment

load_dotenv("../.env")


ALPHAADVANTAGE_API_KEY = os.environ.get("ALPHAADVANTAGE_API_KEY")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET= os.environ.get("REDDIT_API_KEY")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_KEY_SECRET = os.environ.get("TWITTER_API_KEY_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")


def normalize_to_ist(dt_str):
    ist = gettz("Asia/Kolkata")

    # Handle ambiguous or unrecognized timezone names
    tzinfos = {
        "IST": ist,       # India Standard Time
        "GMT": tzutc(),   # Greenwich Mean Time
        "UTC": tzutc(),   # UTC
        "Z": tzutc(),     # Zulu time (Z)
    }

    try:
        # Parse with timezone info
        dt = date_parser.parse(dt_str, tzinfos=tzinfos)

        # If datetime is naive, localize it to UTC before converting
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ist)

        # Convert to IST
        dt_ist = dt.astimezone(ist)
        return dt_ist

    except Exception as e:
        print(f"Error parsing: {dt_str} — {e}")
        return None

headers = {"User-Agent": "Mozilla/5.0"}

class NewsExtractor:
    # Define function to fetch news
    def get_market_news_newsapi(self,company_name,count=100):
        api = NewsApiClient(api_key=NEWS_API_KEY)
        res = api.get_everything(q=company_name, language='en', sort_by='relevancy')
        news = []
        for n in res['articles'][:count]:
            news.append({"Title":n['title'],"Summary":n['description'], "Time":normalize_to_ist(n['publishedAt']), "Link":n['url']})
        return pd.DataFrame(news)
        
    def get_market_news_alphaad(self,ticker):
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol={ticker}&apikey={ALPHAADVANTAGE_API_KEY}'
        r = requests.get(url)
        print(url)
        if r.status_code != 200:
            return []
        articles = r.json()['feed']
        news = []
        for article in articles:
            title = article['title']
            time_published = article['time_published']
            summary = article['summary']
            overall_sentiment_score = article['overall_sentiment_score']
            overall_sentiment_label = article['overall_sentiment_label']
            news.append({"Title":title,"Time":normalize_to_ist(time_published),"Summary":summary,"SentimentScore":overall_sentiment_score,"SentimantLabel":overall_sentiment_label})
        return pd.DataFrame(news)
    
    def get_market_news_x(self,keyword, count=100):
        auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY, TWITTER_API_KEY_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth)
        
        # Search for recent tweets with the keyword
        tweets = api.search_tweets(q=keyword, count=count, lang='en')
        return tweets
    
    def get_market_news_reddit(self,subreddit_name, keyword, limit=50):
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
        subreddit = reddit.subreddit(subreddit_name)
    
        news = []
        for submission in subreddit.search(keyword, limit=limit):
            title = submission.title
            summary = submission.selftext
            link = submission.url
            pub_time = normalize_to_ist(submission.created_utc)
            news.append({"Title":title,"Summary":"", "Time":pub_time, "Link":link})
        return pd.DataFrame(news)
    
    def get_market_news_economictimes(self,url="https://economictimes.indiatimes.com/markets", count=5):
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
        articles = soup.select("ul#just_in li")
        news = []
        for a in articles:
            link_tag = a.find("a")
            title = link_tag['title']
            link = urljoin(url, link_tag["href"])
            time_tag = a.find("time")
            time = normalize_to_ist(time_tag["data-time"])
            news.append({"Title":title,"Summary":"", "Time":time, "Link":link})
        return pd.DataFrame(news)
    
    def get_market_news_pulsezerodha(self,url="https://pulse.zerodha.com/", count=5):
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
        articles = soup.select("ul#news li.box.item")
        news = []
        for article in articles[:count]:
            a = article.select_one("a")
            title = a.get_text(strip=True)
            link = a["href"]
            body = article.select_one("div.desc").get_text(strip=True)
            date_str = article.select_one("span.date")['title']
            pub_time = normalize_to_ist(date_str)
            news.append({"Title":title,"Summary":body,"Link": link,"Time": pub_time})
        return pd.DataFrame(news)
    
    def get_market_news_livemint(self,url="https://www.livemint.com/market/stock-market-news", count=5):
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
        articles = soup.select("div.listtostory")[:count]
        news = []
        for article in articles:
            title = article.find("h2").get_text(strip=True)
            body = ""
            link = urljoin(url, article.find("a")["href"])
            pub_time = normalize_to_ist(article.find('span', class_='fl date').find('span', {'data-updatedtime': True})['data-updatedtime'])
            news.append({"Title":title,"Summary":body,"Link": link,"Time": pub_time})
        return pd.DataFrame(news)
    
    def get_market_news_financialexpress(self,url="https://www.financialexpress.com/market/", count=5):
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
        articles = soup.select("article")
        news = []
        cnt = 0
        for article in articles:
            if cnt>count:
                break
            try:
                a = article.find("a")
                link = a["href"]
                title = a['aria-label']
                body = article.find("p").get_text(strip=True)
                pub_time = normalize_to_ist(article.find("time")['datetime']) 
                news.append({"Title":title,"Summary":body, "Time":pub_time, "Link":link})
                cnt+=1
            except Exception as e:
                continue
        return pd.DataFrame(news)
        
    def get_market_news(self,site,count=5):
        pass
    
    def get_market_news_moneycontrol(self,url="https://www.moneycontrol.com/news/business/markets/" , count=5):
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
        articles = soup.select("li.clearfix")[:count]
        news = []
        for article in articles:
            a_tag = article.find("a")
            if not a_tag: continue
            title = a_tag.get("title", "").strip()
            body = article.find("p").get_text(strip=True) if article.find("p") else ""
            link = a_tag["href"]
            # pub_time = article.find("span").get_text(strip=True)
            pattern = r'\b([A-Za-z]{3,9} \d{2}, \d{4} \d{2}:\d{2} [APap][Mm] [A-Za-z]{2,4})\b'
            matches = re.findall(pattern, str(soup))
            pub_time= normalize_to_ist(matches[0]) 
            news.append({"Title":title,"Summary":body, "Time":pub_time, "Link":link})
        return pd.DataFrame(news)
        
    # Function to get Google News
    def get_market_news_googlenews(self,count=5):
        rss_url = "https://news.google.com/rss/search?q=stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        # Parse the RSS feed
        feed = feedparser.parse(rss_url)
        
        # Extract the news items
        news = []
        for entry in feed.entries[:count]:
            news.append({"Title":entry.title,"Summary":"","Link": entry.link,"Time": normalize_to_ist(entry.published)})
        return pd.DataFrame(news)

        
    def get_market_news_all(self,display=False,unique=True,latest=True):
        news = []
        news.append(self.get_market_news_financialexpress())
        news.append(self.get_market_news_googlenews())
        news.append(self.get_market_news_economictimes())
        news.append(self.get_market_news_livemint())
        news.append(self.get_market_news_moneycontrol())
        news.append(self.get_market_news_pulsezerodha())
        # news.append(self.get_market_news_newsapi(company_name))
        # news.append(self.get_market_news_reddit(company_name,'stock'))
        df = pd.concat(news)
        if latest:
            df = df.sort_values(by="Time",ascending=False)
        if unique:
            df = df.drop_duplicates(subset='Time')
        return df.reset_index(drop=True)


    def fetch_company_news(self,company_name, max_per_site=30):
        sites = [
        "https://economictimes.indiatimes.com/markets",
        "https://www.moneycontrol.com/news/business/markets/",
        "https://www.business-standard.com/markets/stock-market-news",
        "https://www.livemint.com/market/stock-market-news",
        "https://www.financialexpress.com/market/",
        "https://in.investing.com/news/stock-market-news",
        "https://www.nseindia.com/market-data/live-equity-market",
        "https://pulse.zerodha.com/",
        "https://www.cnbctv18.com/market/",
        "https://www.tradingview.com/news/",
        "https://gocharting.com/docs/analytics/news-section",
        "https://www.screener.in/screens/38248/stock-in-quick-news/",
        "https://tradingeconomics.com/stream",
        "https://www.zeebiz.com/markets/stocks",
        "https://www.ndtvprofit.com/markets",
        "https://www.equitymaster.com/research-it/sector-info/stocks.asp",
        "https://www.businesstoday.in/markets",
        "https://www.thehindubusinessline.com/markets/",
        "https://www.republicworld.com/business/",
        "https://www.news18.com/business/",
        "https://www.indiainfoline.com/markets",
        "https://www.bqprime.com/markets"]
        
        all_news = []
        ist = gettz("Asia/Kolkata")
    
        for site in sites:
            # Build the Google News RSS query
            query = f'"{company_name}" site:{site}'.replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={query}+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)
    
            for entry in feed.entries[:max_per_site]:
                try:
                    published = date_parser.parse(entry.published)
                except Exception:
                    published = None
                    
                all_news.append({
                    "company": company_name,
                    "title": entry.title,
                    "link": entry.link,
                    "published": published.astimezone(ist),
                    "source": site
                })
        
        # Sort news by latest date
        return pd.DataFrame(all_news).sort_values(by="published",ascending=False).reset_index(drop=True)


class StockSelector:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
        
    def get_sentiment_scores(self,item):
        return self.analyzer.polarity_scores(str(item.Title) + " " + str(item.Summary))
        
    def parse_llm_output(self,llm_response: str) -> dict:
        """
        Parse LLM YAML-style output for sentiment analysis and stock recommendation.
    
        Args:
            llm_response (str): LLM's raw response in YAML format.
    
        Returns:
            dict: Parsed dictionary with sentiment info and company actions.
        """
        try:
            data = yaml.safe_load(llm_response)
            return data
        except yaml.YAMLError as e:
            # print("YAML parsing error:", e)
            return {}
            
    def add_new_sentiments(self,news_df):
        news_df[['Neg', 'Neu', 'Pos', 'Compound']] = news_df.apply(lambda row: pd.Series(self.get_sentiment_scores(row)), axis=1)
        
    def get_stock_info_from_news(self,news_df,max_count=3):
        top_companies = []
        count = 0
        i=0
        while (count < max_count):
            title = news_df.iloc[i].Title
            description = news_df.iloc[i].Summary
            url = news_df.iloc[i].Link
            
            prompt=f"""You are tasked with analyzing the sentiment of a news article. Extract the main content of the article from the title, description, or, if necessary, from the URL provided. Based on this content:
            1. **Determine the sentiment**: Positive, Neutral, or Negative.
            2. **Assign a sentiment score** between -1.0 (very negative) and +1.0 (very positive).
            3. **Provide reasoning**: A brief explanation for the assigned sentiment.
            4. **Extract key phrases** from the article that support the sentiment and reasoning.
            
            Additionally, identify any companies or stocks mentioned in the article.
            
            For each company mentioned:
            - **Sentiment**: Positive, Neutral, or Negative.
            - **Sentiment Score**: A float between -1.0 and +1.0.
            - **Reasoning**: A brief justification for the sentiment assigned to the company.
            - **Key Phrases**: Relevant quotes or phrases from the article that support the sentiment about the company.
            
            **Input:**
            - **Title**: {title}
            - **Description**: {description}
            - **URL**: {url}
            
            **Output Format**:
            
            ```yaml
            CompaniesCount: [Number of companies mentioned]
            CompaniesData:
              - Name: [Company]
                Sentiment: Positive | Neutral | Negative
                SentimentScore: [Float between -1.0 to 1.0]
                Reasoning: [Short explanation of sentiment]
                KeyPhrases:
                  - "[Key phrase 1]"
                  - "[Key phrase 2]"
            
            **Important Notes:**
            - Ensure that all lists (like Key Phrases) are indented correctly with hyphens (`-`), and all text that may contain special characters (like apostrophes or colons) is enclosed in double quotes (`"`).
            - Maintain the correct YAML syntax, with consistent indentation (2 spaces per indentation level).
            - If no companies are mentioned in the article, set **Companies Count** to 0 and omit the **Companies Data** section from the YAML output.
              
            Example of a well-formed YAML output:
            
            CompaniesCount: 1
            CompaniesData:
              - Name: Nike
                Sentiment: Negative
                SentimentScore: -0.8
                Reasoning: The company's involvement in the controversy has resulted in a loss of trust among its consumers.
                KeyPhrases:
                  - "Nike's alleged rug pull"
                  - "consumer trust plummets"
            
            If no companies are mentioned, the output should look like this:
           
            Companies Count: 0
            
            """
            res = self.llm.invoke(input=prompt).content
            # print(res)
            output = self.parse_llm_output(res)
            
            if ('CompaniesCount' in output) and (output['CompaniesCount'] > 0):
                for company in output['CompaniesData']:
                    top_companies.append({'Name':company['Name'],'Sentiment':company['Sentiment'],'SentimentScore':company['SentimentScore'],'Reasoning':company['Reasoning']})
                count+=1
            i+=1
        return pd.DataFrame(top_companies)
        
class TechnicalInfoExtractor:
    # Define function to fetch financials
    def get_financials(self,ticker):
        stock_data = yf.Ticker(ticker)
        financials = stock_data.financials
        return financials
    
    def get_options_flow(self,symbol, api_key):
        """
        Fetches options flow data for a given stock symbol using Tradier API.
    
        Parameters:
        - symbol (str): Stock symbol (e.g., 'AAPL')
        - api_key (str): Tradier API key
        
        Returns:
        - dict: Options flow data (calls, puts, volumes)
        """
        url = f"https://api.tradier.com/v1/markets/options/chains"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        params = {'symbol': symbol, 'expiration': '2023-05-19'}  # Example expiration date
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        return data
