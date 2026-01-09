import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

# Title
st.title("📈 Stock Price Forecasting Dashboard")
st.markdown("Real-time forecasting using Exponential Smoothing")

# Sidebar
st.sidebar.header("Forecast Parameters")
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

# Load historical data
@st.cache_data
def load_historical_data():
    df = pd.read_csv('data/stock_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

historical_data = load_historical_data()

# Get forecast from API
@st.cache_data(ttl=60)
def get_forecast(periods):
    try:
        response = requests.post(
            "http://localhost:8000/forecast",
            json={"periods": periods}
        )
        return response.json()
    except:
        return None

# Get metrics from API
@st.cache_data
def get_metrics():
    try:
        response = requests.get("http://localhost:8000/metrics")
        return response.json()
    except:
        return None

# Main layout
col1, col2, col3, col4 = st.columns(4)

metrics = get_metrics()
if metrics:
    col1.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    col2.metric("RMSE", f"${metrics['RMSE']:.2f}")
    col3.metric("MAE", f"${metrics['MAE']:.2f}")
    col4.metric("Model", metrics['model_type'])

# Get forecast
forecast_data = get_forecast(forecast_days)

if forecast_data:
    # Create visualization
    fig = go.Figure()
    
    # Show last 90 days of historical + forecast
    # This ensures smooth transition
    recent_history = historical_data.tail(90)
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=recent_history['Date'],
        y=recent_history['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    forecast_dates = pd.to_datetime(forecast_data['dates'])
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['predictions'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['upper_bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['lower_bound'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255,0,0,0.2)',
        fill='tonexty',
        name='95% Confidence',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Stock Price Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.subheader("📊 Forecast Details")
    forecast_df = pd.DataFrame({
        'Date': forecast_data['dates'],
        'Predicted Price': [f"${x:.2f}" for x in forecast_data['predictions']],
        'Lower Bound': [f"${x:.2f}" for x in forecast_data['lower_bound']],
        'Upper Bound': [f"${x:.2f}" for x in forecast_data['upper_bound']]
    })
    st.dataframe(forecast_df, use_container_width=True, height=300)

else:
    st.error("⚠️ Could not connect to API. Make sure it's running on port 8000!")
    st.code("python src/api.py")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust the forecast horizon using the slider above")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown("""
- **Algorithm**: Exponential Smoothing
- **Seasonality**: Weekly (7 days)
- **Trend**: Additive
- **Training Data**: 800 days
""")
