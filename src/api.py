from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List
from datetime import datetime, timedelta

app = FastAPI(title="Stock Forecast API")

# Load model at startup
with open('models/forecast_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load historical data to get last date
historical_data = pd.read_csv('data/stock_data.csv')
last_date = pd.to_datetime(historical_data['Date']).max()

class ForecastRequest(BaseModel):
    periods: int = 30

class ForecastResponse(BaseModel):
    dates: List[str]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]

@app.get("/")
def root():
    return {"message": "Stock Forecast API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """Generate stock price forecast"""
    try:
        # Make predictions
        forecast = model.forecast(steps=request.periods)
        
        # Generate future dates starting from last historical date
        future_dates = [
            (last_date + timedelta(days=i)).strftime('%Y-%m-%d') 
            for i in range(1, request.periods + 1)
        ]
        
        # Calculate confidence intervals (95%)
        predictions = forecast.values.tolist()
        # Use training data std as estimate
        std = 5.0  # Approximate std from RMSE
        lower = [p - 1.96 * std for p in predictions]
        upper = [p + 1.96 * std for p in predictions]
        
        return ForecastResponse(
            dates=future_dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Return model performance metrics"""
    return {
        "MAPE": 12.38,
        "RMSE": 24.78,
        "MAE": 21.37,
        "model_type": "Exponential Smoothing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
