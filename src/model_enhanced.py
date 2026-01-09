import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Train/test split
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

print(f"Training data: {train['Date'].min()} to {train['Date'].max()}")
print(f"Last price: ${train['Close'].iloc[-1]:.2f}")

# ========== MODEL 1: Exponential Smoothing ==========
print("\n🔵 Training Exponential Smoothing...")
es_model = ExponentialSmoothing(
    train['Close'], 
    seasonal_periods=7,
    trend='add',
    seasonal='add'
).fit()

# ========== MODEL 2: ARIMA ==========
print("🟢 Training ARIMA(5,1,2)...")
arima_model = ARIMA(train['Close'], order=(5,1,2)).fit()

# ========== MODEL 3: Prophet ==========
print("🟣 Training Prophet...")
prophet_df = train[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
prophet_model.fit(prophet_df)

# Calculate metrics on test set
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(model, model_name, test_size):
    if model_name == 'Prophet':
        future = model.make_future_dataframe(periods=test_size)
        forecast = model.predict(future)
        preds = forecast['yhat'].tail(test_size).values
    elif model_name == 'ARIMA':
        preds = model.forecast(steps=test_size)
    else:  # ES
        preds = model.forecast(steps=test_size)

    mae = mean_absolute_error(test['Close'], preds)
    rmse = np.sqrt(mean_squared_error(test['Close'], preds))
    mape = np.mean(np.abs((test['Close'] - preds) / test['Close'])) * 100

    print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

es_metrics = calculate_metrics(es_model, 'ES', len(test))
arima_metrics = calculate_metrics(arima_model, 'ARIMA', len(test))
prophet_metrics = calculate_metrics(prophet_model, 'Prophet', len(test))

# Save all models
models = {
    'exponential_smoothing': {'model': es_model, 'metrics': es_metrics},
    'arima': {'model': arima_model, 'metrics': arima_metrics},
    'prophet': {'model': prophet_model, 'metrics': prophet_metrics},
    'last_date': df['Date'].max(),
    'last_price': df['Close'].iloc[-1]
}

with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("\n✅ All models saved to models/all_models.pkl")
