import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pickle
import os

class StockForecaster:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.forecast = None
        
    def load_data(self, csv_path='data/stock_data.csv'):
        """Load stock data from CSV"""
        data = pd.read_csv(csv_path)
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['Date']),
            'Close': data['Close']
        })
        df.set_index('Date', inplace=True)
        return df
    
    def train_test_split(self, df, test_size=0.2):
        """Split data into train/test"""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test
    
    def train(self, train_data):
        """Train Exponential Smoothing model"""
        self.model = ExponentialSmoothing(
            train_data['Close'],
            seasonal_periods=7,
            trend='add',
            seasonal='add'
        )
        self.fitted_model = self.model.fit()
        return self.fitted_model
    
    def predict(self, periods=30):
        """Make predictions"""
        forecast = self.fitted_model.forecast(steps=periods)
        self.forecast = forecast
        return forecast
    
    def evaluate(self, test_data):
        """Calculate metrics"""
        forecast = self.predict(periods=len(test_data))
        
        y_true = test_data['Close'].values
        y_pred = forecast.values
        
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}
    
    def save_model(self, filepath='models/forecast_model.pkl'):
        """Save model"""
        os.makedirs('models', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_model, f)

if __name__ == '__main__':
    forecaster = StockForecaster()
    
    print("📊 Loading data...")
    df = forecaster.load_data()
    print(f"✅ Loaded {len(df)} days")
    
    print("\n🔪 Train/test split...")
    train, test = forecaster.train_test_split(df)
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    print("\n🤖 Training model...")
    forecaster.train(train)
    print("✅ Done!")
    
    print("\n📈 Evaluating...")
    metrics = forecaster.evaluate(test)
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAE: ${metrics['MAE']:.2f}")
    
    forecaster.save_model()
    print("\n✅ All done!")
