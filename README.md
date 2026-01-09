# 📈 Stock Forecast ML Dashboard

Multi-model time series forecasting system with FastAPI backend and Streamlit frontend. Built with production-grade architecture for real-world deployment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)

## 🎯 Features

- **Multiple ML Models**: Exponential Smoothing (Holt-Winters) and ARIMA(5,1,2)
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **Interactive Dashboard**: Real-time forecasting with Streamlit
- **Confidence Intervals**: 95% prediction intervals for all forecasts
- **Model Comparison**: Switch between models and compare performance metrics
- **Responsive Design**: Mobile-friendly interface with Plotly visualizations

## 🏗️ Architecture

```
stock-forecast-ml-dashboard/
├── data/
│   └── stock_data.csv          # Historical stock prices
├── models/
│   └── all_models.pkl          # Trained ML models
├── src/
│   ├── model_fixed.py          # Model training pipeline
│   ├── api_fixed.py            # FastAPI backend
│   └── dashboard_fixed.py      # Streamlit frontend
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Donald8585/stock-forecast-ml-dashboard.git
cd stock-forecast-ml-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
```bash
python src/model_fixed.py
```

### 4. Start API Server
```bash
python src/api_fixed.py
```

### 5. Launch Dashboard (New Terminal)
```bash
streamlit run src/dashboard_fixed.py
```

Visit `http://localhost:8501` to see the dashboard!

## 📊 Models

### Exponential Smoothing (Holt-Winters)
- **Type**: Additive trend + Additive seasonality
- **Seasonality**: 7 days (weekly pattern)
- **Best for**: Stable trends with regular patterns
- **Performance**: MAPE ~12%, RMSE ~$25

### ARIMA(5,1,2)
- **Parameters**: 
  - AR(5): 5 autoregressive lags
  - I(1): First-order differencing
  - MA(2): 2 moving average terms
- **Best for**: Stationary time series
- **Performance**: MAPE ~13%, RMSE ~$26

## 🔌 API Endpoints

### `POST /forecast`
Generate stock price forecast

**Request Body:**
```json
{
  "periods": 30,
  "model": "exponential_smoothing"
}
```

**Response:**
```json
{
  "dates": ["2026-01-09", "2026-01-10", ...],
  "predictions": [184.5, 185.2, ...],
  "lower_bound": [182.3, 183.0, ...],
  "upper_bound": [186.7, 187.4, ...],
  "model_used": "exponential_smoothing",
  "last_historical_date": "2026-01-08",
  "last_historical_price": 186.19
}
```

### `GET /models`
List available models with metrics

### `GET /metrics/{model_name}`
Get specific model performance metrics

### API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI

## 🎨 Dashboard Features

1. **Model Selection**: Choose between ES and ARIMA models
2. **Forecast Horizon**: Adjust prediction period (7-90 days)
3. **Performance Metrics**: Real-time MAPE, RMSE, and MAE
4. **Interactive Charts**: Zoom, pan, and hover for details
5. **Forecast Table**: Detailed predictions with confidence intervals
6. **Key Statistics**: Price change and percentage change indicators

## 🐳 Docker Deployment

```bash
# Build image
docker build -t stock-forecast .

# Run container
docker run -p 8000:8000 -p 8501:8501 stock-forecast
```

## 📈 Results

| Model | MAPE | RMSE | MAE |
|-------|------|------|-----|
| Exponential Smoothing | 12.38% | $24.78 | $21.37 |
| ARIMA(5,1,2) | 13.12% | $26.45 | $22.89 |

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **ML Libraries**: Statsmodels, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Python**: 3.8+

## 📝 Future Enhancements

- [ ] Real-time data integration with Yahoo Finance API
- [ ] Multiple stock ticker support
- [ ] LSTM/Transformer models
- [ ] Automated retraining pipeline
- [ ] Authentication and user management
- [ ] Historical forecast accuracy tracking
- [ ] Export forecasts to CSV/Excel

## 👨‍💻 Author

**Alfred So**
- LinkedIn: [linkedin.com/in/alfred-so](https://www.linkedin.com/in/alfred-so/)
- GitHub: [github.com/Donald8585](https://github.com/Donald8585/)
- Email: fiverrkroft@gmail.com

## 📄 License

MIT License - feel free to use this project for learning and portfolio purposes!

## 🙏 Acknowledgments

Built as part of ML Engineering portfolio for data science roles in San Francisco Bay Area.
