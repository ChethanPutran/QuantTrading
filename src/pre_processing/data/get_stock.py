from transformers import pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import praw
import tweepy
import pandas_ta as ta
import requests
from textblob import TextBlob
import yfinance as yf

# Set up Twitter API keys (you need to get your own from Twitter Developer Portal)
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate to the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Fetch stock-related tweets for a particular stock (e.g., AAPL)


# Define the stock and the news API key
stock_symbol = 'AAPL'
api_key = 'YOUR_NEWS_API_KEY'  # Replace with your NewsAPI key

# Function to fetch news articles for a stock


def get_news(stock_symbol, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']

    return articles

# Function to perform sentiment analysis on news articles


def analyze_sentiment(articles):
    positive, negative, neutral = 0, 0, 0

    for article in articles:
        # Analyze sentiment of article title and description
        text = article['title'] + " " + article['description']
        sentiment = TextBlob(text).sentiment.polarity

        if sentiment > 0:
            positive += 1
        elif sentiment < 0:
            negative += 1
        else:
            neutral += 1

    return positive, negative, neutral


# Fetch and analyze news
articles = get_news(stock_symbol, api_key)
positive, negative, neutral = analyze_sentiment(articles)

# Display sentiment analysis results
print(f"Positive news: {positive}")
print(f"Negative news: {negative}")
print(f"Neutral news: {neutral}")


# Fetch stock data (use yfinance or any other API)

def fetch_stock_data(stock_symbol, interval="5m", period="1d"):
    stock_data = yf.download(stock_symbol, interval=interval, period=period)
    stock_data["RSI"] = ta.rsi(
        stock_data["Close"], length=14)  # Adding RSI indicator
    stock_data["EMA_9"] = ta.ema(
        stock_data["Close"], length=9)  # Adding EMA indicator
    return stock_data

# Function to select stocks based on sentiment


def select_stocks_based_on_sentiment(stock_symbol, sentiment_score):
    # Define threshold for sentiment-based decision
    if sentiment_score > 0.2:
        return "BUY"
    elif sentiment_score < -0.2:
        return "SELL"
    else:
        return "HOLD"

# Combine sentiment with stock data for final decision


def stock_selection(stock_symbol, api_key):
    # Get news and sentiment
    articles = get_news(stock_symbol, api_key)
    positive, negative, neutral = analyze_sentiment(articles)

    sentiment_score = (positive - negative) / (positive +
                                               negative + neutral)  # Calculate overall sentiment

    # Fetch stock data (e.g., price, RSI, EMA)
    stock_data = fetch_stock_data(stock_symbol)

    # Get the latest price and technical indicators
    latest_close = stock_data['Close'].iloc[-1]
    latest_rsi = stock_data['RSI'].iloc[-1]
    latest_ema = stock_data['EMA_9'].iloc[-1]

    # Determine the action based on sentiment and technical indicators
    sentiment_action = select_stocks_based_on_sentiment(
        stock_symbol, sentiment_score)

    print(f"Stock: {stock_symbol}")
    print(f"Sentiment: {sentiment_score:.2f} -> Action: {sentiment_action}")
    print(
        f"Latest Close: {latest_close}, RSI: {latest_rsi}, EMA_9: {latest_ema}")

    # Apply technical indicator strategy (example: RSI < 30 -> Buy, EMA crossover -> Buy/Sell)
    if latest_rsi < 30 and sentiment_action == "BUY":
        print("Additional Action: Consider buying based on RSI.")
    if latest_ema < latest_close and sentiment_action == "BUY":
        print("Additional Action: Buy based on EMA crossover.")

    return sentiment_action


# Run the stock selection for a specific stock
action = stock_selection('AAPL', 'YOUR_NEWS_API_KEY')



def fetch_tweets(stock_symbol, count=100):
    tweets = api.search(q=stock_symbol, count=count,
                        lang="en", result_type="recent")
    tweet_data = []
    for tweet in tweets:
        tweet_data.append(tweet.text)
    return tweet_data


# Example usage
stock_symbol = 'AAPL'
tweets = fetch_tweets(stock_symbol)
print(tweets[:5])  # Print first 5 tweets

# Set up Reddit API credentials
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

# Fetch Reddit posts related to a stock


def fetch_reddit_posts(stock_symbol, limit=100):
    posts = reddit.subreddit('stocks').search(stock_symbol, limit=limit)
    post_data = []
    for post in posts:
        post_data.append(post.title + " " + post.selftext)
    return post_data


# Example usage
reddit_posts = fetch_reddit_posts('AAPL')
print(reddit_posts[:5])  # Print first 5 Reddit posts


# Example data
data = ["Stock is going up!", "The market crashed today",
        "Positive news for AAPL", "Bearish trend for TSLA"]
labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Create a pipeline for text classification
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

# Predict sentiment of new texts
new_texts = ["Bullish sentiment for AAPL", "TSLA is facing challenges"]
predictions = model.predict(new_texts)
print(predictions)  # 1 = Positive, 0 = Negative


# Load pre-trained BERT model for sentiment-analysis
sentiment_pipeline = pipeline("sentiment-analysis")

# Example text for sentiment analysis
texts = [
    "AAPL reports great earnings, stock price will go up!",
    "The market is crashing today, avoid stocks like TSLA.",
]

# Predict sentiment using BERT
predictions = sentiment_pipeline(texts)
for text, prediction in zip(texts, predictions):
    print(
        f"Text: {text}\nSentiment: {prediction['label']} with score {prediction['score']:.2f}")


def aggregate_sentiment(stock_symbol):
    # Fetch and analyze news from multiple sources (Twitter, Reddit)
    twitter_sentiment = analyze_sentiment(
        fetch_tweets(stock_symbol))  # Get Twitter sentiment
    reddit_sentiment = analyze_sentiment(
        fetch_reddit_posts(stock_symbol))  # Get Reddit sentiment
    bert_sentiment = sentiment_pipeline(
        [stock_symbol])[0]  # Get BERT sentiment

    # Aggregate sentiment from all sources
    total_positive = twitter_sentiment[0] + reddit_sentiment[0] + \
        (1 if bert_sentiment['label'] == 'POSITIVE' else 0)
    total_negative = twitter_sentiment[1] + reddit_sentiment[1] + \
        (1 if bert_sentiment['label'] == 'NEGATIVE' else 0)
    sentiment_score = (total_positive - total_negative) / \
        (total_positive + total_negative)  # Normalize score

    return sentiment_score

# Example of stock selection based on combined sentiment


def stock_selection(stock_symbol):
    sentiment_score = aggregate_sentiment(stock_symbol)

    if sentiment_score > 0.2:
        return "BUY"
    elif sentiment_score < -0.2:
        return "SELL"
    else:
        return "HOLD"


# Run selection for a stock
action = stock_selection("AAPL")
print(f"Action for AAPL: {action}")
