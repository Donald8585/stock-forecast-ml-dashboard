from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List, Optional
from datetime import timedelta

app = FastAPI(title="Enhanced Stock Forecast API")

# Load models at startup
with open('models/all_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

# Load historical data
historical_data = pd.read_csv('data/stock_data.csv')
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
last_date = models_data['last_date']
last_price = models_data['last_price']

class ForecastRequest(BaseModel):
    periods: int = 30
    model: Optional[str] = 'exponential_smoothing'  # arima, prophet, exponential_smoothing

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
        "message": "Enhanced Stock Forecast API",
        "models": ["exponential_smoothing", "arima", "prophet"],
        "status": "running"
    }

@app.get("/models")
def list_models():
    return {
        "available_models": [
            {
                "name": "exponential_smoothing",
                "metrics": models_data['exponential_smoothing']['metrics']
            },
            {
                "name": "arima",
                "metrics": models_data['arima']['metrics']
            },
            {
                "name": "prophet",
                "metrics": models_data['prophet']['metrics']
            }
        ]
    }

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        model_name = request.model
        if model_name not in models_data:
            raise HTTPException(400, f"Model {model_name} not found")

        model_obj = models_data[model_name]['model']

        # Generate predictions
        if model_name == 'prophet':
            future = model_obj.make_future_dataframe(periods=request.periods)
            forecast_df = model_obj.predict(future)
            predictions = forecast_df['yhat'].tail(request.periods).values.tolist()
            lower = forecast_df['yhat_lower'].tail(request.periods).values.tolist()
            upper = forecast_df['yhat_upper'].tail(request.periods).values.tolist()
        else:  # ARIMA or ES
            predictions = model_obj.forecast(steps=request.periods).tolist()
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
    if model_name not in models_data:
        raise HTTPException(404, f"Model {model_name} not found")
    return models_data[model_name]['metrics']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
