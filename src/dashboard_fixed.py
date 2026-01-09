import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Stock Forecast Pro", layout="wide", page_icon="📈")

st.title("📈 Stock Price Forecasting Dashboard Pro")
st.markdown("Multi-Model Time Series Forecasting | Exponential Smoothing & ARIMA")

# Sidebar
st.sidebar.header("🎛️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["exponential_smoothing", "arima"],
    format_func=lambda x: {
        "exponential_smoothing": "📊 Exponential Smoothing (Holt-Winters)",
        "arima": "📉 ARIMA(5,1,2)"
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

# Get model metrics
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
    col2.metric("📉 MAPE", f"{metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error")
    col3.metric("📐 RMSE", f"${metrics['RMSE']:.2f}", help="Root Mean Squared Error")
    col4.metric("📏 MAE", f"${metrics['MAE']:.2f}", help="Mean Absolute Error")

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
        name='Historical Data',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>Historical</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Add connecting point (last historical point) - THIS FIXES THE GAP!
    last_hist_date = pd.to_datetime(forecast_data['last_historical_date'])
    last_hist_price = forecast_data['last_historical_price']

    fig.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[last_hist_price],
        mode='markers',
        name='Last Known Price',
        marker=dict(color='#2ca02c', size=12, symbol='circle', line=dict(color='white', width=2)),
        hovertemplate='<b>Last Known</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Forecast line
    forecast_dates = pd.to_datetime(forecast_data['dates'])

    # Connect last historical to first forecast with a bridge line
    bridge_dates = [last_hist_date, forecast_dates[0]]
    bridge_prices = [last_hist_price, forecast_data['predictions'][0]]

    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_prices,
        mode='lines',
        name='Bridge',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_data['predictions'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2.5, dash='dash'),
        hovertemplate='<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
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
        name='95% Confidence Interval',
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
        st.subheader("📊 Detailed Forecast Table")
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
        price_change = forecast_data['predictions'][-1] - last_hist_price
        pct_change = (price_change / last_hist_price) * 100

        st.metric("Current Price", f"${last_hist_price:.2f}", help="Last known historical price")
        st.metric("Avg Forecast Price", f"${avg_price:.2f}")
        st.metric(
            f"{forecast_days}-Day Change", 
            f"${price_change:.2f}", 
            f"{pct_change:+.2f}%",
            delta_color="normal"
        )

        st.markdown("---")
        st.markdown(f"**📅 Last Historical:** {forecast_data['last_historical_date']}")
        st.markdown(f"**🚀 Forecast Start:** {forecast_data['dates'][0]}")
        st.markdown(f"**🏁 Forecast End:** {forecast_data['dates'][-1]}")

else:
    st.error("⚠️ Could not connect to API. Make sure it's running!")
    st.code("python src/api_fixed.py", language="bash")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
if model_choice == "exponential_smoothing":
    st.sidebar.markdown('''
    **Exponential Smoothing (Holt-Winters)**
    - ✅ Trend: Additive
    - ✅ Seasonality: Weekly (7 days)
    - ✅ Best for: Stable trends with seasonality
    - ⚙️ Smoothing parameters auto-optimized
    ''')
else:
    st.sidebar.markdown('''
    **ARIMA(5,1,2)**
    - 📊 AR: 5 autoregressive lags
    - 📉 I: 1st order differencing
    - 📈 MA: 2 moving average terms
    - ✅ Best for: Stationary time series
    ''')

st.sidebar.markdown("---")
st.sidebar.success("🚀 **Tech Stack:** FastAPI + Streamlit + Statsmodels")
st.sidebar.info("💡 Adjust horizon slider to see different forecasts")

ticker = st.sidebar.selectbox("Stock Ticker", ["AAPL", "TSLA", "NVDA", "GOOGL"])

