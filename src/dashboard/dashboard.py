"""

**dashboard frontend**
To visualize stock predictions, historical data, and sentiment analysis results.

For this, we’ll use **Streamlit**, a simple Python-based dashboard framework, which is perfect for visualizing real-time data and predictions.
**Set Up the Dashboard Structure**

Let’s create a simple Streamlit dashboard that can:

* Display real-time predictions for a selected stock.
* Show historical predictions from the SQLite database.
* Visualize sentiment scores from Twitter/Reddit.

### File Structure:

```
dashboard/
├── app.py                  # Streamlit dashboard code
├── requirements.txt        # List of dependencies
└── README.md               # Project overview
```
### **Streamlit Requirements (`requirements.txt`)**

This file will include all the dependencies for your project.

```
streamlit
pandas
sqlite3
tweepy
vaderSentiment
praw
```

"""


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

### 📊 4. **Advanced Visualizations (Optional)**

Once the basic dashboard is running, you can add more visualizations like:

* **Stock Price Graphs**: Using `plotly` or `matplotlib` to display historical stock prices (from an API like Yahoo Finance).
* **Sentiment over Time**: Plotting sentiment scores (Twitter/Reddit) over time alongside stock price to visualize correlations.

**Example Visualization Code (for stock price with Plotly)**:

```python
import plotly.graph_objects as go

# Example DataFrame with stock prices
df = pd.DataFrame({'date': ['2021-01-01', '2021-01-02'], 'price': [150, 155]})
fig = go.Figure([go.Scatter(x=df['date'], y=df['price'], mode='lines', name='Stock Price')])
st.plotly_chart(fig)
```

---

### 🚀 Next Steps

1. **Run the Streamlit dashboard**: Test the interface for inputting tickers and displaying predictions.
2. **Add more features**: Sentiment over time, options flow visualization, historical comparison.
3. **Deploy**: Deploy the app to services like **Streamlit Cloud**, **Heroku**, or **AWS**.

Would you like to proceed with adding visualizations or deploying the app? Let me know if you need help with any specific step.

# Add Options Flow Data (You could show options flow graph here if required)
# Example: options_flow = get_options_flow(ticker)

# ### 📋 3. **Running the Dashboard**

# # To run the dashboard, you just need to execute the following command:
# streamlit run dashboard/app.py