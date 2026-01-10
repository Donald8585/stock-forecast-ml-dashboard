import yfinance as yf
import pandas as pd
import numpy as np
from lstm_model import StockLSTM
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Popular stocks to pre-train
stocks = ['GOOGL', 'MSFT', 'TSLA', 'NVDA']

for ticker in stocks:
    print(f"\n{'='*50}")
    print(f"Training {ticker}...")
    print(f"{'='*50}")
    
    # Download 5 years of data
    df = yf.download(ticker, period='5y', progress=False)
    prices = df['Close'].values
    
    # Initialize LSTM
    lstm = StockLSTM()
    
    # Prepare data
    X, y, _ = lstm.prepare_data(prices)
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    
    # Train model
    print(f"Training on {len(X_train)} samples...")
    lstm.train(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Save model
    filepath = f'models/{ticker}_lstm.h5'
    lstm.save_model(filepath)
    print(f"✅ Saved to {filepath}")

print("\n✅ All models trained!")
