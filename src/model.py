import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pickle
import os

class StockForecaster:
    def __init__(self):
        self.model = None
        self.forecast = None
        
    def load_data(self, csv_path='data/stock_data.csv'):
        """Load stock data from CSV"""
        if not os.path.exists(csv_path):
            csv_path = 'stock_data.csv'
        
        data = pd.read_csv(csv_path)
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['Date']),
            'y': data['Close']
        })
        return df
    
    def train_test_split(self, df, test_size=0.2):
        """Split data into train/test"""
        split_idx = int(len(df) * (1 - test_size))
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
        self.forecast = forecast
        return forecast
    
    def evaluate(self, test_data):
        """Calculate metrics"""
        forecast = self.predict(test_data)
        
        y_true = test_data['y'].values
        y_pred = forecast['yhat1'].values
        
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        
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

if __name__ == '__main__':
    forecaster = StockForecaster()
    
    print("📊 Loading data...")
    df = forecaster.load_data()
    print(f"✅ Loaded {len(df)} days of data")
    
    print("\n🔪 Splitting train/test...")
    train, test = forecaster.train_test_split(df)
    print(f"Train: {len(train)} days, Test: {len(test)} days")
    
    print("\n🤖 Training NeuralProphet model...")
    forecaster.train(train)
    print("✅ Training complete!")
    
    print("\n📈 Evaluating model...")
    metrics = forecaster.evaluate(test)
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: ${metrics['RMSE']:.2f}")
    print(f"MAE: ${metrics['MAE']:.2f}")
    
    print("\n💾 Saving model...")
    forecaster.save_model()
    print("✅ Model saved!")
