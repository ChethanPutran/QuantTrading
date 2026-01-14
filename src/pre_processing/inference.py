
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def evaulate(trades):
    ticker = yf.Ticker("AAPL")
    opt_symbol = ticker.options[-1]  # Last expiry
    opt_chain = ticker.option_chain(opt_symbol)

    # For example, get a specific call option
    option_df = opt_chain.calls
    chosen_option = option_df.iloc[0]['contractSymbol']
    option = yf.Ticker(chosen_option)

    opt_hist = option.history(period="2y")
    opt_hist = opt_hist[['Close']]
    opt_hist.columns = ['price']

    portfolio = pd.DataFrame(index=opt_hist.index)
    portfolio['position'] = 0
    portfolio['cash'] = 100000  # Starting capital
    portfolio['value'] = portfolio['cash']

    for _, trade in trades.iterrows():
        trade_day = pd.Timestamp(trade['date'])
        if trade_day in portfolio.index:
            qty = trade['qty']
            price = trade['price'] * 100  # Options are typically 100x
            if trade['type'] == 'buy_call':
                portfolio.loc[trade_day:, 'position'] += qty
                portfolio.loc[trade_day:, 'cash'] -= price * qty
            elif trade['type'] == 'sell_call':
                portfolio.loc[trade_day:, 'position'] -= qty
                portfolio.loc[trade_day:, 'cash'] += price * qty

    # Final value
    portfolio['option_price'] = opt_hist['price'] * 100
    portfolio['value'] = portfolio['cash'] + portfolio['position'] * portfolio['option_price']
    portfolio['returns'] = portfolio['value'].pct_change().fillna(0)



    cumulative_pnl = portfolio['value'].iloc[-1] - portfolio['value'].iloc[0]
    print(f"Cumulative PnL: ${cumulative_pnl:.2f}")


    sharpe_ratio = (portfolio['returns'].mean() / portfolio['returns'].std()) * (252**0.5)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")


    rolling_max = portfolio['value'].cummax()
    drawdown = portfolio['value'] / rolling_max - 1
    max_drawdown = drawdown.min()
    print(f"Max Drawdown: {max_drawdown:.2%}")



    # Example placeholder
    greeks_df = pd.DataFrame(index=portfolio.index)
    greeks_df['delta'] = 0.5  # fill with real values
    greeks_df['gamma'] = 0.02

    portfolio['delta_exposure'] = greeks_df['delta'] * portfolio['position']
    portfolio['gamma_exposure'] = greeks_df['gamma'] * portfolio['position']


    plt.figure(figsize=(14, 5))
    plt.plot(portfolio.index, portfolio['delta_exposure'], label="Delta Exposure")
    plt.plot(portfolio.index, portfolio['gamma_exposure'], label="Gamma Exposure")
    plt.title("Delta & Gamma Exposure Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    trades = pd.DataFrame([
    {'date': '2023-01-02', 'type': 'buy_call', 'option': 'AAPL230120C00150000', 'price': 5.2, 'qty': 1},
    {'date': '2023-01-08', 'type': 'sell_call', 'option': 'AAPL230120C00150000', 'price': 7.5, 'qty': 1}
])
    trades['date'] = pd.to_datetime(trades['date'])