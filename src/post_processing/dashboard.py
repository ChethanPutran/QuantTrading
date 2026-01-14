import streamlit as st
import pandas as pd
import sqlite3
from app.predict import run_prediction_for_all_tickers  # Import from your prediction module
from app.database import get_prediction_history  # Import from your SQLite functions
from app.sentiment import get_twitter_sentiment, get_reddit_sentiment

# Database Connection
def get_db_connection():
    conn = sqlite3.connect('db/predictions.db')
    return conn

# Display Header
st.title('Stock Prediction Dashboard')

# Stock Ticker Input
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL):', 'AAPL')

# Display Sentiment Analysis Data
if st.button('Get Sentiment'):
    sentiment_twitter = get_twitter_sentiment(ticker, api_key='your_key', api_secret='your_secret', access_token='your_token', access_token_secret='your_token_secret')
    sentiment_reddit = get_reddit_sentiment(ticker, "stocks", client_id="your_id", client_secret="your_secret", user_agent="your_agent")
    
    st.write(f"Twitter Sentiment Score for {ticker}: {sentiment_twitter}")
    st.write(f"Reddit Sentiment Score for {ticker}: {sentiment_reddit}")

# Display Real-Time Prediction
if st.button(f'Get Prediction for {ticker}'):
    prediction = run_prediction_for_all_tickers(ticker_override=ticker)  # Get the stock prediction using your model
    st.write(f'Predicted Price for {ticker}: {prediction["predicted_price"]}')

# Display Historical Predictions
if st.button(f'Show Historical Data for {ticker}'):
    conn = get_db_connection()
    query = f"SELECT * FROM predictions WHERE ticker = '{ticker}'"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        st.write(f"No historical data available for {ticker}.")
    else:
        st.write(f"Historical Predictions for {ticker}")
        st.write(df)
