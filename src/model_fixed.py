import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
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

print(f"📊 Training data: {train['Date'].min()} to {train['Date'].max()}")
print(f"💰 Last training price: ${train['Close'].iloc[-1]:.2f}")
print(f"📈 Test size: {len(test)} days\n")

# ========== MODEL 1: Exponential Smoothing ==========
print("🔵 Training Exponential Smoothing...")
es_model = ExponentialSmoothing(
    train['Close'], 
    seasonal_periods=7,
    trend='add',
    seasonal='add'
).fit()

# ========== MODEL 2: ARIMA ==========
print("🟢 Training ARIMA(5,1,2)...")
arima_model = ARIMA(train['Close'], order=(5,1,2)).fit()

# Calculate metrics on test set
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(model, model_name, test_data):
    preds = model.forecast(steps=len(test_data))

    mae = mean_absolute_error(test_data, preds)
    rmse = np.sqrt(mean_squared_error(test_data, preds))
    mape = np.mean(np.abs((test_data - preds) / test_data)) * 100

    print(f"  ✓ MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}

print("\n📊 Model Performance on Test Set:")
es_metrics = calculate_metrics(es_model, 'ES', test['Close'])
arima_metrics = calculate_metrics(arima_model, 'ARIMA', test['Close'])

# Retrain on ALL data for final models
print("\n🔄 Retraining on full dataset...")
es_final = ExponentialSmoothing(
    df['Close'], 
    seasonal_periods=7,
    trend='add',
    seasonal='add'
).fit()

arima_final = ARIMA(df['Close'], order=(5,1,2)).fit()

# Save all models
models = {
    'exponential_smoothing': {'model': es_final, 'metrics': es_metrics},
    'arima': {'model': arima_final, 'metrics': arima_metrics},
    'last_date': df['Date'].max(),
    'last_price': df['Close'].iloc[-1]
}

with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("\n✅ Models saved to models/all_models.pkl")
print(f"📅 Last date: {models['last_date']}")
print(f"💵 Last price: ${models['last_price']:.2f}")
