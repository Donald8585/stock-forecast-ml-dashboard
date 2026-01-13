import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
import os
warnings.filterwarnings("ignore")

# Import LSTM model
try:
    from lstm_model import StockLSTM
except ImportError:
    StockLSTM = None

# Twelve Data API Configuration
TWELVE_DATA_API_KEY = "bbf951d474d549a2be99d3bb594b2327"

# Page config
st.set_page_config(
    page_title="StockForecast - AI Time Series Prediction",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(120deg, #1e3a8a, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-header {
    text-align: center;
    color: #64748b;
    font-size: 1.2rem;
    margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 StockForecast</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LSTM + Prophet Time Series Forecasting</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    if StockLSTM is not None:
        model_type = st.selectbox("Forecasting Model", 
                                 ["LSTM (Deep Learning)", "Prophet (Meta)"], 
                                 index=1,
                                 help="LSTM: Deep learning. Prophet: Statistical forecasting.")
    else:
        model_type = "Prophet (Meta)"
        st.info("LSTM model not available. Using Prophet.")
    
    # Stock ticker
    ticker = st.text_input("Stock Ticker", 
                          value="AAPL",
                          help="Enter stock symbol (e.g., AAPL, MSFT, IBM, TSLA)")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    st.markdown("#### 📅 Data Range")
    start = st.date_input("Start Date", value=start_date)
    end = st.date_input("End Date", value=end_date)
    
    # Forecast period
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
    
    # Train button
    train_button = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Model info
    st.info("""
**📊 Model Info**

**Prophet (Meta)** ⭐ Recommended
- Time series forecasting
- Yearly & Weekly seasonality
- ~20 second training
- Production-ready

**LSTM (Deep Learning)**
- Lightning-fast training (~20 seconds)
- Custom model per stock ticker
- 3-layer LSTM architecture
""")
    
    st.markdown("#### 🔥 Try These Tickers")
    st.markdown("`AAPL` `MSFT` `GOOGL` `TSLA`")
    
    st.markdown("#### 🔗 Links")
    st.markdown("- [GitHub](https://github.com/Donald8585/)")
    st.markdown("- [LinkedIn](https://www.linkedin.com/in/alfred-so/)")

# Twelve Data API fetcher
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_twelve_data(ticker_symbol, start_date, end_date):
    """Fetch stock data from Twelve Data API"""
    try:
        # Calculate date range
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Construct URL
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': ticker_symbol,
            'interval': '1day',
            'start_date': start_str,
            'end_date': end_str,
            'apikey': TWELVE_DATA_API_KEY,
            'format': 'JSON'
        }
        
        # Make request
        st.info(f"📡 Fetching {ticker_symbol} data...")
        response = requests.get(url, params=params, timeout=20)
        data = response.json()
        
        # Error checking
        if 'status' in data and data['status'] == 'error':
            st.error(f"❌ API Error: {data.get('message', 'Unknown error')}")
            return None, data.get('message', 'API error')
        
        if 'code' in data and data['code'] == 429:
            st.warning("⚠️ Rate limit reached. Please wait a minute.")
            return None, "Rate limit exceeded"
        
        if 'values' not in data:
            st.error("❌ No data available for this ticker")
            return None, "No data available"
        
        # Convert to DataFrame
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        
        if len(df) < 100:
            st.warning(f"⚠️ Only {len(df)} days of data available (need 100+)")
            return None, f"Insufficient data: {len(df)} days"
        
        st.success(f"✅ Fetched {len(df)} days of data!")
        return df, None
        
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. Please try again.")
        return None, "Request timeout"
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, str(e)

# Load pretrained LSTM
@st.cache_resource
def load_pretrained_lstm(ticker):
    """Load pre-trained LSTM model if available"""
    if StockLSTM is None:
        return None, False
    
    filepath = f"models/{ticker}_lstm.h5"
    if os.path.exists(filepath):
        lstm = StockLSTM()
        lstm.load_model(filepath)
        return lstm, True
    
    return None, False

# Main content
if train_button:
    try:
        # Download data
        with st.spinner(f"📡 Fetching {ticker} data from Twelve Data..."):
            df, error = fetch_twelve_data(ticker, start, end)
        
        if df is None or error:
            st.warning(f"⚠️ Switching to Demo Mode")
            st.info(f"Reason: {error}")
            
            # Generate realistic synthetic demo data
            dates = pd.date_range(start=start, end=end, freq='D')
            base_price = np.random.uniform(100, 300)
            returns = np.random.randn(len(dates)) * 0.015  # 1.5% daily volatility
            prices = base_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({'Close': prices}, index=dates)
            df['Open'] = df['Close'] * np.random.uniform(0.99, 1.01, len(df))
            df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.0, 1.02, len(df))
            df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.98, 1.0, len(df))
            df['Volume'] = np.random.randint(5000000, 20000000, len(df))
            
            st.success(f"🎭 Generated {len(df)} days of synthetic {ticker} data")
        
        # Prepare data
        df_clean = df.copy()
        prices = df_clean['Close'].values
        dates = pd.to_datetime(df_clean.index)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stock", ticker.upper())
        
        with col2:
            current_price = prices[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col3:
            price_change = prices[-1] - prices[-2]
            pct_change = (price_change / prices[-2]) * 100
            st.metric("Daily Change", f"{pct_change:.2f}%", delta=f"${price_change:.2f}")
        
        with col4:
            st.metric("Data Points", len(prices))
        
        st.markdown("---")
        
        # Model training
        if "LSTM" in model_type and StockLSTM is not None:
            # LSTM Model
            lstm_model, is_pretrained = load_pretrained_lstm(ticker)
            
            if is_pretrained:
                st.success(f"✅ Using cached {ticker} model!")
                with st.spinner(f"🔮 Forecasting next {forecast_days} days..."):
                    predictions = lstm_model.predict_future(prices, forecast_days)
                st.success("Forecast complete!")
            else:
                with st.spinner(f"🧠 Training LSTM model for {ticker}... (~20 seconds)"):
                    progress_bar = st.progress(0)
                    
                    lstm_model = StockLSTM()
                    X, y, _ = lstm_model.prepare_data(prices)
                    split = int(0.8 * len(X))
                    X_train, y_train = X[:split], y[:split]
                    X_test, y_test = X[split:], y[split:]
                    
                    progress_bar.progress(20)
                    lstm_model.train(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                    progress_bar.progress(80)
                    
                    predictions = lstm_model.predict_future(prices, forecast_days)
                    progress_bar.progress(100)
                    st.success("Model training complete!")
            
            lower_bound = predictions * 0.95
            upper_bound = predictions * 1.05
            
            # Calculate metrics
            X, y, _ = lstm_model.prepare_data(prices)
            split = int(0.8 * len(X))
            X_test, y_test = X[split:], y[split:]
            y_pred = lstm_model.model.predict(X_test, verbose=0)
            y_pred = lstm_model.scaler.inverse_transform(y_pred)
            y_test_inv = lstm_model.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mape = mean_absolute_percentage_error(y_test_inv, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
            mae = np.mean(np.abs(y_test_inv - y_pred))
        
        else:
            # Prophet Model
            df_prophet = pd.DataFrame({'ds': dates, 'y': prices})
            if df_prophet['ds'].dt.tz is not None:
                df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
            
            split_idx = int(len(df_prophet) * 0.8)
            train_data = df_prophet[:split_idx].copy()
            test_data = df_prophet[split_idx:].copy()
            
            with st.spinner("🔮 Training Prophet model... (~20 seconds)"):
                progress_bar = st.progress(0)
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                model.fit(train_data)
                progress_bar.progress(100)
                st.success("Prophet model trained!")
            
            with st.spinner(f"📊 Forecasting next {forecast_days} days..."):
                last_date = df_prophet['ds'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                future_df = pd.DataFrame({'ds': future_dates})
                forecast = model.predict(future_df)
                
                predictions = forecast['yhat'].values
                lower_bound = forecast['yhat_lower'].values
                upper_bound = forecast['yhat_upper'].values
            
            # Calculate metrics
            test_forecast = model.predict(test_data)
            y_true = test_data['y'].values
            y_pred = test_forecast['yhat'].values
            
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = np.mean(np.abs(y_true - y_pred))
        
        # Display metrics
        st.markdown("### 📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div style="text-align: center; padding: 20px; background: #f0f9ff; border-radius: 10px"><h3 style="color: #0369a1; margin: 0">MAPE</h3><h2 style="margin: 5px 0">{mape:.2f}%</h2></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div style="text-align: center; padding: 20px; background: #f0fdf4; border-radius: 10px"><h3 style="color: #15803d; margin: 0">RMSE</h3><h2 style="margin: 5px 0">${rmse:.2f}</h2></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div style="text-align: center; padding: 20px; background: #fef3c7; border-radius: 10px"><h3 style="color: #a16207; margin: 0">MAE</h3><h2 style="margin: 5px 0">${mae:.2f}</h2></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Plot forecast
        st.markdown(f"### 📈 {ticker} Stock Price Forecast")
        
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines',
            name='Historical',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates, y=predictions,
            mode='lines',
            name='Forecast',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates, y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=lower_bound,
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download forecast
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predictions,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        })
        
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Fatal Error: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b">
<p><strong>Built by Alfred So | ML Engineer</strong></p>
<p>🎓 AWS ML • GCP ML • Azure AI • Databricks ML • NVIDIA AIIO</p>
<p><small>📊 Twelve Data API: 800 calls/day | Data cached 1 hour</small></p>
</div>
""", unsafe_allow_html=True)
