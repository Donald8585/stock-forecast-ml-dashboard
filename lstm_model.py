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
        self.look_back = 60  # Use 60 days to predict next day
        
    def prepare_data(self, data, look_back=60):
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
        """Predict future prices"""
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        # Get last look_back days
        last_sequence = scaled_data[-self.look_back:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            # Reshape for prediction
            X_test = current_sequence.reshape(1, self.look_back, 1)
            
            # Predict next day
            next_pred = self.model.predict(X_test, verbose=0)[0][0]
            predictions.append(next_pred)
            
            # Update sequence (sliding window)
            current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
        # Inverse transform to get actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
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
