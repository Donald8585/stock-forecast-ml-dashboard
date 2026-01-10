import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker='AAPL', period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.to_csv('data/stock_data.csv', index=False)
    print(f"✅ Fetched {len(df)} days of {ticker} data")
    return df

if __name__ == "__main__":
    fetch_stock_data('AAPL', '5y')  # Apple, 5 years
