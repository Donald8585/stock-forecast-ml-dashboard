import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Stock Forecast Pro", layout="wide", page_icon="📈")

st.title("📈 Stock Price Forecasting Dashboard Pro")
st.markdown("Real-Time Stock Forecasting | Exponential Smoothing & ARIMA")

# Sidebar
st.sidebar.header("🎛️ Configuration")

# Stock ticker selection
ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META", "NFLX"],
    format_func=lambda x: {
        "AAPL": "🍎 Apple (AAPL)",
        "TSLA": "🚗 Tesla (TSLA)",
        "NVDA": "🎮 NVIDIA (NVDA)",
        "GOOGL": "🔍 Google (GOOGL)",
        "MSFT": "💻 Microsoft (MSFT)",
        "AMZN": "📦 Amazon (AMZN)",
        "META": "📘 Meta (META)",
        "NFLX": "🎬 Netflix (NFLX)"
    }[x]
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["exponential_smoothing", "arima"],
    format_func=lambda x: {
        "exponential_smoothing": "📊 Exponential Smoothing",
        "arima": "📉 ARIMA(5,1,2)"
    }[x]
)

forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)

# Load real stock data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_real_data(ticker_symbol):
    try:
        with st.spinner(f"Fetching {ticker_symbol} data from Yahoo Finance..."):
            # Get 5 years of data
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="5y")

            if df.empty:
                st.error(f"No data found for {ticker_symbol}")
                return None

            df = df.reset_index()
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Date'] = pd.to_datetime(df['Date'])

            return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Train model
@st.cache_resource
def train_model(model_type, data, ticker_symbol):
    with st.spinner(f"Training {model_type} model for {ticker_symbol}..."):
        try:
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
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return None

# Load data
df = load_real_data(ticker)

if df is not None and len(df) > 100:
    last_date = df['Date'].max()
    last_price = df['Close'].iloc[-1]

    # Train model
    model = train_model(model_choice, df, ticker)

    if model is not None:
        # Generate forecast
        forecast = model.forecast(steps=forecast_days)
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

        # Calculate confidence intervals
        std_estimate = df['Close'].std() * 0.1  # Dynamic based on volatility
        lower_bound = forecast - 1.96 * std_estimate
        upper_bound = forecast + 1.96 * std_estimate

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        # Calculate price change
        prev_price = df['Close'].iloc[-2]
        price_change = last_price - prev_price
        price_change_pct = (price_change / prev_price) * 100

        col1.metric("📊 Stock", ticker)
        col2.metric("💰 Last Price", f"${last_price:.2f}", f"{price_change_pct:+.2f}%")
        col3.metric("📅 Last Date", last_date.strftime('%Y-%m-%d'))
        col4.metric("🔮 Model", model_choice.replace('_', ' ').title())
        col5.metric("📈 Forecast Days", forecast_days)

        st.markdown("---")

        # Create visualization
        fig = go.Figure()

        # Historical data (last 180 days)
        recent_history = df.tail(180)

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
            title=f'{ticker} Stock Price Forecast - {model_choice.replace("_", " ").title()}',
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
                'Upper Bound': [f"${x:.2f}" for x in upper_bound],
                'Change from Today': [f"${x - last_price:+.2f}" for x in forecast]
            })
            st.dataframe(forecast_df.head(14), use_container_width=True, height=400)

        with col2:
            st.subheader("📈 Key Statistics")
            avg_price = forecast.mean()
            price_change_forecast = forecast.iloc[-1] - last_price
            pct_change_forecast = (price_change_forecast / last_price) * 100

            st.metric("Current Price", f"${last_price:.2f}")
            st.metric("Avg Forecast Price", f"${avg_price:.2f}")
            st.metric(
                f"{forecast_days}-Day Forecast", 
                f"${forecast.iloc[-1]:.2f}",
                f"{pct_change_forecast:+.2f}%",
                delta_color="normal"
            )

            # Additional stats
            st.markdown("---")
            st.markdown(f"**52-Week High:** ${df['Close'].tail(252).max():.2f}")
            st.markdown(f"**52-Week Low:** ${df['Close'].tail(252).min():.2f}")
            st.markdown(f"**Volatility (Std):** ${df['Close'].std():.2f}")
            st.markdown(f"**Last Updated:** {last_date.strftime('%Y-%m-%d')}")

            # Trend indicator
            trend = "📈 Bullish" if pct_change_forecast > 0 else "📉 Bearish"
            st.markdown(f"**Forecast Trend:** {trend}")

else:
    st.error("⚠️ Unable to load stock data. Please try again or select a different ticker.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
if model_choice == "exponential_smoothing":
    st.sidebar.markdown('''
    **Exponential Smoothing**
    - ✅ Trend: Additive
    - ✅ Seasonality: Weekly (7 days)
    - ✅ Best for: Stable trends
    - ⚙️ Auto-optimized parameters
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
st.sidebar.info("💡 Data refreshes every hour")
st.sidebar.success("🚀 **Powered by:** Yahoo Finance")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **By:** Alfred So")
st.sidebar.markdown("🔗 [GitHub](https://github.com/Donald8585)")
st.sidebar.markdown("💼 [LinkedIn](https://linkedin.com/in/alfred-so)")

# Disclaimer
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Disclaimer:** This is for educational purposes only. Not financial advice.")
