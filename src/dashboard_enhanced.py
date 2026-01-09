import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="Stock Forecast Dashboard Pro", layout="wide", page_icon="📈")

# Custom CSS
st.markdown('''<style>
.big-metric {font-size: 24px; font-weight: bold; color: #1f77b4;}
</style>''', unsafe_allow_html=True)

st.title("📈 Stock Price Forecasting Dashboard Pro")
st.markdown("Multi-Model Time Series Forecasting with Real-Time API")

# Sidebar
st.sidebar.header("🎛️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["exponential_smoothing", "arima", "prophet"],
    format_func=lambda x: {
        "exponential_smoothing": "📊 Exponential Smoothing",
        "arima": "📉 ARIMA(5,1,2)",
        "prophet": "🔮 Facebook Prophet"
    }[x]
)

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
def get_forecast(periods, model):
    try:
        response = requests.post(
            "http://localhost:8000/forecast",
            json={"periods": periods, "model": model},
            timeout=30
        )
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Get model metrics from API
@st.cache_data
def get_model_metrics(model_name):
    try:
        response = requests.get(f"http://localhost:8000/metrics/{model_name}")
        return response.json()
    except:
        return None

# Display metrics
metrics = get_model_metrics(model_choice)
if metrics:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Model", model_choice.replace('_', ' ').title())
    col2.metric("📉 MAPE", f"{metrics['MAPE']:.2f}%")
    col3.metric("📐 RMSE", f"${metrics['RMSE']:.2f}")
    col4.metric("📏 MAE", f"${metrics['MAE']:.2f}")

st.markdown("---")

# Get forecast
forecast_data = get_forecast(forecast_days, model_choice)

if forecast_data:
    # Create visualization
    fig = go.Figure()

    # Show last 90 days of historical
    recent_history = historical_data.tail(90)

    # Historical data
    fig.add_trace(go.Scatter(
        x=recent_history['Date'],
        y=recent_history['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Historical</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Add connecting point (last historical point)
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(forecast_data['last_historical_date'])],
        y=[forecast_data['last_historical_price']],
        mode='markers',
        name='Last Known',
        marker=dict(color='green', size=10, symbol='circle'),
        hovertemplate='<b>Last Known</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Forecast
    forecast_dates = pd.to_datetime(forecast_data['dates'])
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['predictions'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
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
        fillcolor='rgba(255,127,14,0.2)',
        fill='tonexty',
        name='95% Confidence',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f'Stock Price Forecast - {model_choice.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=550,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast details
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Forecast Table")
        forecast_df = pd.DataFrame({
            'Date': forecast_data['dates'],
            'Predicted Price': [f"${x:.2f}" for x in forecast_data['predictions']],
            'Lower Bound': [f"${x:.2f}" for x in forecast_data['lower_bound']],
            'Upper Bound': [f"${x:.2f}" for x in forecast_data['upper_bound']]
        })
        st.dataframe(forecast_df.head(14), use_container_width=True, height=400)

    with col2:
        st.subheader("📈 Key Statistics")
        avg_price = sum(forecast_data['predictions']) / len(forecast_data['predictions'])
        price_change = forecast_data['predictions'][-1] - forecast_data['last_historical_price']
        pct_change = (price_change / forecast_data['last_historical_price']) * 100

        st.metric("Current Price", f"${forecast_data['last_historical_price']:.2f}")
        st.metric("Avg Forecast Price", f"${avg_price:.2f}")
        st.metric(f"Forecast Change ({forecast_days}d)", f"${price_change:.2f}", f"{pct_change:+.2f}%")

        st.markdown(f"**Last Historical Date:** {forecast_data['last_historical_date']}")
        st.markdown(f"**Forecast Start:** {forecast_data['dates'][0]}")
        st.markdown(f"**Forecast End:** {forecast_data['dates'][-1]}")

else:
    st.error("⚠️ Could not connect to API. Make sure it's running!")
    st.code("python src/api_enhanced.py", language="bash")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
if model_choice == "exponential_smoothing":
    st.sidebar.markdown('''
    **Exponential Smoothing**
    - Trend: Additive
    - Seasonality: Weekly (7 days)
    - Best for: Stable trends
    ''')
elif model_choice == "arima":
    st.sidebar.markdown('''
    **ARIMA(5,1,2)**
    - AR: 5 lags
    - I: 1 differencing
    - MA: 2 terms
    - Best for: Stationary data
    ''')
else:
    st.sidebar.markdown('''
    **Facebook Prophet**
    - Daily & Weekly seasonality
    - Automatic trend detection
    - Best for: Complex patterns
    ''')

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Built with:** FastAPI + Streamlit + Prophet")
