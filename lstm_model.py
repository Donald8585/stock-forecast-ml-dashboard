import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os

class StockLSTM:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 20  # ✅ Changed from 60 to 20
        
    def prepare_data(self, data, look_back=20):  # ✅ Changed default
        """Prepare data for LSTM training"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data
    
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, verbose=0):
        """Train the model"""
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.1
        )
        return history
    
    def predict_future(self, data, forecast_days=30):
        """Predict future prices with strong trend preservation"""
        
        # Calculate recent trend (last 10 days)
        recent_prices = data[-10:]
        trend_slope = (recent_prices[-1] - recent_prices[0]) / 10
        
        # Get LSTM predictions
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        last_sequence = scaled_data[-self.look_back:]
        
        lstm_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            X_test = current_sequence.reshape(1, self.look_back, 1)
            next_pred = self.model.predict(X_test, verbose=0)[0][0]
            lstm_predictions.append(next_pred)
            current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
        # Inverse transform LSTM predictions
        lstm_predictions = np.array(lstm_predictions).reshape(-1, 1)
        lstm_predictions = self.scaler.inverse_transform(lstm_predictions)
        lstm_predictions = lstm_predictions.flatten()
        
        # ✅ STRONG FIX: Use trend continuation instead of broken LSTM
        last_price = data[-1]
        trend_predictions = []
        
        for i in range(forecast_days):
            # Continue recent trend with slight dampening
            trend_pred = last_price + (trend_slope * (i + 1) * 0.5)  # 50% dampening
            
            # Limit to ±2% daily change
            if i > 0:
                max_change = trend_predictions[i-1] * 0.02
                trend_pred = np.clip(trend_pred, 
                                    trend_predictions[i-1] - max_change,
                                    trend_predictions[i-1] + max_change)
            
            trend_predictions.append(trend_pred)
        
        trend_predictions = np.array(trend_predictions)
        
        # Blend: 40% LSTM, 60% Trend (favor trend over broken LSTM)
        final_predictions = lstm_predictions * 0.4 + trend_predictions * 0.6
        
        return final_predictions

    
    def save_model(self, filepath):
        """Save model and scaler"""
        self.model.save(filepath)
        # Save scaler separately
        import joblib
        joblib.dump(self.scaler, filepath.replace('.h5', '_scaler.pkl'))
    
    def load_model(self, filepath):
        """Load pre-trained model and scaler"""
        self.model = keras.models.load_model(filepath)
        # Load scaler
        import joblib
        self.scaler = joblib.load(filepath.replace('.h5', '_scaler.pkl'))
