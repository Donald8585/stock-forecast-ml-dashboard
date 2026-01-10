import pandas as pd
import numpy as np
<<<<<<< HEAD
from neuralprophet import NeuralProphet
=======
from statsmodels.tsa.holtwinters import ExponentialSmoothing
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pickle
import os

class StockForecaster:
    def __init__(self):
        self.model = None
<<<<<<< HEAD
=======
        self.fitted_model = None
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
        self.forecast = None
        
    def load_data(self, csv_path='data/stock_data.csv'):
        """Load stock data from CSV"""
<<<<<<< HEAD
        if not os.path.exists(csv_path):
            csv_path = 'stock_data.csv'
        
        data = pd.read_csv(csv_path)
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['Date']),
            'y': data['Close']
        })
=======
        data = pd.read_csv(csv_path)
        df = pd.DataFrame({
            'Date': pd.to_datetime(data['Date']),
            'Close': data['Close']
        })
        df.set_index('Date', inplace=True)
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
        return df
    
    def train_test_split(self, df, test_size=0.2):
        """Split data into train/test"""
        split_idx = int(len(df) * (1 - test_size))
<<<<<<< HEAD
        train = df[:split_idx].copy()
        test = df[split_idx:].copy()
        return train, test
    
    def train(self, train_data):
        """Train NeuralProphet model"""
        self.model = NeuralProphet(
            epochs=50,
            learning_rate=0.01,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.model.fit(train_data, freq='D')
        return self.model
    
    def predict(self, test_data):
        """Make predictions"""
        forecast = self.model.predict(test_data)
=======
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
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
        self.forecast = forecast
        return forecast
    
    def evaluate(self, test_data):
        """Calculate metrics"""
<<<<<<< HEAD
        forecast = self.predict(test_data)
        
        y_true = test_data['y'].values
        y_pred = forecast['yhat1'].values
=======
        forecast = self.predict(periods=len(test_data))
        
        y_true = test_data['Close'].values
        y_pred = forecast.values
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
        
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        
<<<<<<< HEAD
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }
    
    def save_model(self, filepath='models/neural_prophet_model.pkl'):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath='models/neural_prophet_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
=======
        return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}
    
    def save_model(self, filepath='models/forecast_model.pkl'):
        """Save model"""
        os.makedirs('models', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_model, f)
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4

if __name__ == '__main__':
    forecaster = StockForecaster()
    
    print("📊 Loading data...")
    df = forecaster.load_data()
<<<<<<< HEAD
    print(f"✅ Loaded {len(df)} days of data")
    
    print("\n🔪 Splitting train/test...")
    train, test = forecaster.train_test_split(df)
    print(f"Train: {len(train)} days, Test: {len(test)} days")
    
    print("\n🤖 Training NeuralProphet model...")
    forecaster.train(train)
    print("✅ Training complete!")
    
    print("\n📈 Evaluating model...")
=======
    print(f"✅ Loaded {len(df)} days")
    
    print("\n🔪 Train/test split...")
    train, test = forecaster.train_test_split(df)
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    print("\n🤖 Training model...")
    forecaster.train(train)
    print("✅ Done!")
    
    print("\n📈 Evaluating...")
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
    metrics = forecaster.evaluate(test)
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAE: ${metrics['MAE']:.2f}")
    
<<<<<<< HEAD
    print("\n💾 Saving model...")
    forecaster.save_model()
    print("✅ Model saved!")
=======
    forecaster.save_model()
    print("\n✅ All done!")
>>>>>>> 749ceb72305e6cc9339a49efbe347074bb2f42d4
