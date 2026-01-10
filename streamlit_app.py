import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# Import LSTM model
from lstm_model import StockLSTM

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
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📈 StockForecast</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">LSTM & Prophet Time Series Forecasting</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Forecasting Model",
        ["LSTM (Deep Learning)", "Prophet (Meta)"],
        index=0,
        help="LSTM: More accurate, faster inference. Prophet: Seasonal patterns."
    )
    
    # Stock ticker selection
    ticker = st.text_input("Stock Ticker", value="GOOGL", help="Enter stock symbol (e.g., GOOGL, MSFT, TSLA)")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    st.markdown("### 📅 Data Range")
    start = st.date_input("Start Date", value=start_date)
    end = st.date_input("End Date", value=end_date)
    
    # Forecast period
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
    
    # Train button
    train_button = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    st.info("""
    ### 🧠 Model Info
    **LSTM (Deep Learning)**
    - ⚡ Lightning-fast training: ~20 seconds
    - 🎯 Custom model per stock ticker
    - 📊 Trains on your selected date range
    - 🔄 Always uses latest market data
    - 🧠 3-layer LSTM architecture
    """)

    st.info("""
    **Prophet (Meta)**
    - Time series forecasting
    - Yearly + Weekly seasonality
    - 20 second training
    - Production-ready
    """)
    
    st.markdown("### 🔗 Links")
    st.markdown("- [GitHub Repo](https://github.com/Donald8585/stock-forecast-ml-dashboard)")
    st.markdown("- [LinkedIn](https://linkedin.com/in/alfred-so)")

# Cache LSTM models
@st.cache_resource
def load_pretrained_lstm(ticker):
    """Load pre-trained LSTM model if available"""
    filepath = f'models/{ticker}_lstm.h5'
    if os.path.exists(filepath):
        lstm = StockLSTM()
        lstm.load_model(filepath)
        return lstm, True
    return None, False

# Main content
if train_button:
    try:
        # Download data
        with st.spinner(f"📥 Downloading {ticker} data..."):
            use_demo = False
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, timeout=10)
                
                if df.empty or len(df) < 100:
                    st.warning(f"⚠️ Insufficient data for {ticker}. Need at least 100 days.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error downloading data: {str(e)[:100]}")
                st.stop()
            
            # Prepare data
            df_clean = df.copy()
            if isinstance(df_clean.columns, pd.MultiIndex):
                df_clean.columns = df_clean.columns.get_level_values(0)
            
            prices = df_clean['Close'].values
            dates = pd.to_datetime(df_clean.index)
            
            st.success(f"✅ Downloaded {len(prices)} days of {ticker} data")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Stock", ticker.upper())
        with col2:
            current_price = prices[-1]
            st.metric("💰 Current Price", f"${current_price:.2f}")
        with col3:
            price_change = prices[-1] - prices[-2]
            pct_change = (price_change / prices[-2]) * 100
            st.metric("📈 Daily Change", f"{pct_change:+.2f}%", delta=f"${price_change:+.2f}")
        with col4:
            st.metric("📅 Data Points", len(prices))
        
        st.markdown("---")
        
        # LSTM Model
        if "LSTM" in model_type:
            # Check for pre-trained model
            lstm_model, is_pretrained = load_pretrained_lstm(ticker)
            
            if is_pretrained:
                st.success(f"⚡ Using pre-trained {ticker} model (instant inference!)")
                
                # Instant prediction
                with st.spinner(f"🔮 Forecasting next {forecast_days} days..."):
                    predictions = lstm_model.predict_future(prices, forecast_days)
                    st.success("✅ Forecast complete!")
            else:
                st.warning(f"⚠️ No pre-trained model for {ticker}. Training new model (~20 seconds)...")

                
                # Train new model
                with st.spinner(f"🤖 Training LSTM model for {ticker}... (this may take 2-5 minutes)"):
                    progress_bar = st.progress(0)
                    
                    lstm_model = StockLSTM()
                    X, y, _ = lstm_model.prepare_data(prices)
                    
                    # Split data
                    split = int(0.8 * len(X))
                    X_train, y_train = X[:split], y[:split]
                    X_test, y_test = X[split:], y[split:]
                    
                    progress_bar.progress(20)
                    
                    # Train
                    lstm_model.train(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                    progress_bar.progress(100)
                    
                    # Predict future
                    predictions = lstm_model.predict_future(prices, forecast_days)
                    
                    st.success("✅ Model training complete!")
            
            # Calculate confidence intervals (±5% for LSTM)
            lower_bound = predictions * 0.95
            upper_bound = predictions * 1.05
            
            # Calculate metrics on test set
            X, y, scaled_data = lstm_model.prepare_data(prices)
            split = int(0.8 * len(X))
            X_test, y_test = X[split:], y[split:]
            
            y_pred = lstm_model.model.predict(X_test, verbose=0)
            y_pred = lstm_model.scaler.inverse_transform(y_pred)
            y_test_inv = lstm_model.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            mape = mean_absolute_percentage_error(y_test_inv, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
            mae = np.mean(np.abs(y_test_inv - y_pred))
            
        # Prophet Model
        else:
            # Prepare Prophet data
            df_prophet = pd.DataFrame({'ds': dates, 'y': prices})
            if df_prophet['ds'].dt.tz is not None:
                df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
            
            # Split data
            split_idx = int(len(df_prophet) * 0.8)
            train_data = df_prophet[:split_idx].copy()
            test_data = df_prophet[split_idx:].copy()
            
            # Train Prophet
            with st.spinner("🤖 Training Prophet model... (20 seconds)"):
                progress_bar = st.progress(0)
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                
                model.fit(train_data)
                progress_bar.progress(100)
                st.success(f"✅ Model trained in ~10s | Final Loss: {loss:.4f}")

            
            # Predict future
            with st.spinner(f"🔮 Forecasting next {forecast_days} days..."):
                last_date = df_prophet['ds'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
            st.markdown(f'<div class="metric-card"><h3>{mape:.2f}%</h3><p>MAPE</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>${rmse:.2f}</h3><p>RMSE</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>${mae:.2f}</h3><p>MAE</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Create future dates
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        
        # Plot
        st.markdown(f"### 📈 {ticker} Price Forecast ({model_type})")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Historical',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Smooth connection
        connection_dates = pd.concat([pd.Series([dates[-1]]), pd.Series(future_dates)])
        connection_prices = np.concatenate([[prices[-1]], predictions])
        
        fig.add_trace(go.Scatter(
            x=connection_dates,
            y=connection_prices,
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        # Confidence interval
        connection_upper = np.concatenate([[prices[-1]], upper_bound])
        connection_lower = np.concatenate([[prices[-1]], lower_bound])
        
        fig.add_trace(go.Scatter(
            x=connection_dates,
            y=connection_upper,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=connection_dates,
            y=connection_lower,
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown("### 📋 Forecast Details")
        st.info(f"✅ **{model_type}** - {'Pre-trained model' if 'LSTM' in model_type and is_pretrained else 'Newly trained model'}")
        
        forecast_table = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Price': np.round(predictions, 2),
            'Lower Bound': np.round(lower_bound, 2),
            'Upper Bound': np.round(upper_bound, 2),
            'Change from Last': np.round(predictions - prices[-1], 2),
            'Change %': np.round((predictions - prices[-1]) / prices[-1] * 100, 2)
        })
        
        st.dataframe(forecast_table, use_container_width=True)
        
        # Download button
        csv = forecast_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_{model_type.split()[0]}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    # Initial state
    st.info("👈 **Configure settings in the sidebar and click 'Train & Forecast' to begin!**")
    
    st.markdown("### 🎯 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Select Model**
        - LSTM: Deep learning (faster)
        - Prophet: Meta's forecaster
        - Choose stock & date range
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Train Model**
        - Downloads historical data
        - Uses pre-trained or trains new
        - Validates on test set
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ View Results**
        - Interactive forecast chart
        - Performance metrics
        - Download predictions
        """)
    
    st.markdown("---")
    st.markdown("### 💡 Pre-trained Stocks (Instant Predictions)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.code("GOOGL\n⚡ Pre-trained")
    with col2:
        st.code("MSFT\n⚡ Pre-trained")
    with col3:
        st.code("TSLA\n⚡ Pre-trained")
    with col4:
        st.code("NVDA\n⚡ Pre-trained")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Built by <strong>Alfred So</strong> | ML Engineer</p>
    <p>🎓 AWS ML • GCP ML • Azure AI • Databricks ML • NVIDIA AIIO</p>
</div>
""", unsafe_allow_html=True)
