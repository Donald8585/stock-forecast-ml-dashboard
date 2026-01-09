import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import numpy as np

st.set_page_config(page_title="Stock Forecast Pro", layout="wide", page_icon="📈")

st.title("📈 Stock Price Forecasting Dashboard")
st.markdown("Multi-Model Time Series Forecasting | Exponential Smoothing & ARIMA")

# Sidebar
st.sidebar.header("🎛️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["exponential_smoothing", "arima"],
    format_func=lambda x: {
        "exponential_smoothing": "📊 Exponential Smoothing",
        "arima": "📉 ARIMA(5,1,2)"
    }[x]
)

forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/stock_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        st.error("Data file not found. Using sample data.")
        # Generate sample data
        import numpy as np
        from datetime import datetime, timedelta

        np.random.seed(42)
        days = 1500
        end_date = datetime(2026, 1, 8)
        start_date = end_date - timedelta(days=days-1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        trend = np.linspace(150, 185, days)
        seasonality = 10 * np.sin(np.linspace(0, 12*np.pi, days))
        noise = np.random.normal(0, 2, days)
        prices = trend + seasonality + noise

        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': prices - np.random.uniform(0, 2, days),
            'High': prices + np.random.uniform(0, 3, days),
            'Low': prices - np.random.uniform(0, 3, days),
            'Volume': np.random.randint(1000000, 10000000, days)
        })
        return df

# Train model
@st.cache_resource
def train_model(model_type, data):
    with st.spinner(f"Training {model_type} model..."):
        if model_type == "exponential_smoothing":
            model = ExponentialSmoothing(
                data['Close'],
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            ).fit()
        else:  # ARIMA
            model = ARIMA(data['Close'], order=(5,1,2)).fit()
    return model

# Load data
df = load_data()
last_date = df['Date'].max()
last_price = df['Close'].iloc[-1]

# Train model
model = train_model(model_choice, df)

# Generate forecast
forecast = model.forecast(steps=forecast_days)
forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

# Calculate confidence intervals
std_estimate = 5.0  # Approximate standard deviation
lower_bound = forecast - 1.96 * std_estimate
upper_bound = forecast + 1.96 * std_estimate

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Model", model_choice.replace('_', ' ').title())
col2.metric("💰 Last Price", f"${last_price:.2f}")
col3.metric("📅 Last Date", last_date.strftime('%Y-%m-%d'))
col4.metric("🔮 Forecast Days", forecast_days)

st.markdown("---")

# Create visualization
fig = go.Figure()

# Historical data (last 90 days)
recent_history = df.tail(90)

fig.add_trace(go.Scatter(
    x=recent_history['Date'],
    y=recent_history['Close'],
    mode='lines',
    name='Historical Data',
    line=dict(color='#1f77b4', width=2.5),
    hovertemplate='<b>Historical</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

# Last known point
fig.add_trace(go.Scatter(
    x=[last_date],
    y=[last_price],
    mode='markers',
    name='Last Known Price',
    marker=dict(color='#2ca02c', size=12, symbol='circle', line=dict(color='white', width=2)),
    hovertemplate='<b>Last Known</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

# Bridge line
bridge_dates = [last_date, forecast_dates[0]]
bridge_prices = [last_price, forecast.iloc[0]]

fig.add_trace(go.Scatter(
    x=bridge_dates,
    y=bridge_prices,
    mode='lines',
    line=dict(color='#ff7f0e', width=2, dash='dot'),
    showlegend=False,
    hoverinfo='skip'
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast,
    mode='lines',
    name='Forecast',
    line=dict(color='#ff7f0e', width=2.5, dash='dash'),
    hovertemplate='<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
))

# Confidence intervals
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=upper_bound,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=lower_bound,
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
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'Predicted Price': [f"${x:.2f}" for x in forecast],
        'Lower Bound': [f"${x:.2f}" for x in lower_bound],
        'Upper Bound': [f"${x:.2f}" for x in upper_bound]
    })
    st.dataframe(forecast_df.head(14), use_container_width=True, height=400)

with col2:
    st.subheader("📈 Key Statistics")
    avg_price = forecast.mean()
    price_change = forecast.iloc[-1] - last_price
    pct_change = (price_change / last_price) * 100

    st.metric("Current Price", f"${last_price:.2f}")
    st.metric("Avg Forecast Price", f"${avg_price:.2f}")
    st.metric(
        f"{forecast_days}-Day Change", 
        f"${price_change:.2f}", 
        f"{pct_change:+.2f}%",
        delta_color="normal"
    )

    st.markdown("---")
    st.markdown(f"**📅 Last Historical:** {last_date.strftime('%Y-%m-%d')}")
    st.markdown(f"**🚀 Forecast Start:** {forecast_dates[0].strftime('%Y-%m-%d')}")
    st.markdown(f"**🏁 Forecast End:** {forecast_dates[-1].strftime('%Y-%m-%d')}")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
if model_choice == "exponential_smoothing":
    st.sidebar.markdown('''
    **Exponential Smoothing**
    - ✅ Trend: Additive
    - ✅ Seasonality: Weekly (7 days)
    - ✅ Best for: Stable trends
    ''')
else:
    st.sidebar.markdown('''
    **ARIMA(5,1,2)**
    - 📊 AR: 5 lags
    - 📉 I: 1st differencing
    - 📈 MA: 2 terms
    - ✅ Best for: Stationary data
    ''')

st.sidebar.markdown("---")
st.sidebar.success("🚀 **Built with:** Streamlit + Statsmodels")
st.sidebar.info("💡 Adjust settings above to see different forecasts")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **By:** Alfred So")
st.sidebar.markdown("🔗 [GitHub](https://github.com/Donald8585)")
st.sidebar.markdown("💼 [LinkedIn](https://linkedin.com/in/alfred-so)")
