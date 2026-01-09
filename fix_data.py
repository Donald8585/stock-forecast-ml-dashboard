
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate data up to Jan 8, 2026
np.random.seed(42)
days = 1500
end_date = datetime(2026, 1, 8)
start_date = end_date - timedelta(days=days-1)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Stock price simulation
trend = np.linspace(150, 185, days)
seasonality = 10 * np.sin(np.linspace(0, 12*np.pi, days))
noise = np.random.normal(0, 2, days)
prices = trend + seasonality + noise

df = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),  # String format to avoid Excel issues
    'Close': prices,
    'Open': prices - np.random.uniform(0, 2, days),
    'High': prices + np.random.uniform(0, 3, days),
    'Low': prices - np.random.uniform(0, 3, days),
    'Volume': np.random.randint(1000000, 10000000, days)
})

# Save
df.to_csv('data/stock_data.csv', index=False)
print(f"✅ Saved {len(df)} rows to data/stock_data.csv")
print(f"First date: {df['Date'].iloc[0]}")
print(f"Last date: {df['Date'].iloc[-1]}")
print(f"Last price: ${df['Close'].iloc[-1]:.2f}")
