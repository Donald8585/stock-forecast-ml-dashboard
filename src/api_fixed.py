from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List, Optional
from datetime import timedelta

app = FastAPI(title="Stock Forecast API Pro")

# Load models at startup
with open('models/all_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

last_date = models_data['last_date']
last_price = models_data['last_price']

print(f"✅ API loaded successfully")
print(f"📅 Last historical date: {last_date}")
print(f"💵 Last historical price: ${last_price:.2f}")

class ForecastRequest(BaseModel):
    periods: int = 30
    model: Optional[str] = 'exponential_smoothing'  # arima, exponential_smoothing

class ForecastResponse(BaseModel):
    dates: List[str]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    model_used: str
    last_historical_date: str
    last_historical_price: float

@app.get("/")
def root():
    return {
        "message": "Stock Forecast API Pro",
        "models": ["exponential_smoothing", "arima"],
        "status": "running",
        "last_date": str(last_date.date())
    }

@app.get("/models")
def list_models():
    return {
        "available_models": [
            {
                "name": "exponential_smoothing",
                "description": "Holt-Winters with weekly seasonality",
                "metrics": models_data['exponential_smoothing']['metrics']
            },
            {
                "name": "arima",
                "description": "ARIMA(5,1,2) model",
                "metrics": models_data['arima']['metrics']
            }
        ]
    }

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        model_name = request.model
        if model_name not in ['exponential_smoothing', 'arima']:
            raise HTTPException(400, f"Model {model_name} not found. Available: exponential_smoothing, arima")

        model_obj = models_data[model_name]['model']

        # Generate predictions
        predictions = model_obj.forecast(steps=request.periods).tolist()

        # Calculate confidence intervals
        std = models_data[model_name]['metrics']['RMSE'] / 2
        lower = [p - 1.96 * std for p in predictions]
        upper = [p + 1.96 * std for p in predictions]

        # Generate future dates starting EXACTLY from last historical date
        future_dates = [
            (last_date + timedelta(days=i)).strftime('%Y-%m-%d') 
            for i in range(1, request.periods + 1)
        ]

        return ForecastResponse(
            dates=future_dates,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_used=model_name,
            last_historical_date=last_date.strftime('%Y-%m-%d'),
            last_historical_price=float(last_price)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{model_name}")
def get_model_metrics(model_name: str):
    if model_name not in ['exponential_smoothing', 'arima']:
        raise HTTPException(404, f"Model {model_name} not found")
    return models_data[model_name]['metrics']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
