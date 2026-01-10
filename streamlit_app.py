import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

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
st.markdown('<p class="subtitle">NeuralProphet Time Series Forecasting System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # Stock ticker selection
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years
    
    st.markdown("### 📅 Data Range")
    start = st.date_input("Start Date", value=start_date)
    end = st.date_input("End Date", value=end_date)
    
    # Forecast period
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=7)
    
    # Train button
    train_button = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.info("""
    **NeuralProphet**
    - Deep learning time series
    - Yearly + Weekly seasonality
    - 50 epochs training
    - Adaptive learning rate
    """)
    
    st.markdown("### 🔗 Links")
    st.markdown("- [GitHub Repo](https://github.com/Donald8585/time-series-forecast-dashboard)")
    st.markdown("- [LinkedIn](https://linkedin.com/in/alfred-so)")

# Main content
if train_button:
    try:
        with st.spinner(f"📥 Downloading {ticker} data..."):
            # Download data
            # Download data with better error handling
try:
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    # Check if download succeeded
    if df.empty or len(df) < 10:
        st.error(f"❌ Failed to download data for {ticker}. Trying alternative method...")
        
        # Alternative: Use Ticker object
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        
        if df.empty:
            st.error(f"❌ No data available for {ticker}. Please try a different ticker (e.g., GOOGL, MSFT, TSLA)")
            st.stop()
            
except Exception as e:
    st.error(f"❌ Download error: {str(e)}")
    st.info("💡 Try: GOOGL, MSFT, TSLA, NVDA, or wait a few seconds and retry")
    st.stop()

            
            if df.empty:
                st.error(f"❌ No data found for {ticker}. Please check the ticker symbol.")
                st.stop()
            
            # Prepare data for NeuralProphet
            df_prophet = pd.DataFrame({
                'ds': df.index,
                'y': df['Close'].values
            }).reset_index(drop=True)
            
            st.success(f"✅ Downloaded {len(df_prophet)} days of {ticker} data")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Stock", ticker.upper())
        with col2:
            current_price = df_prophet['y'].iloc[-1]
            st.metric("💰 Current Price", f"${current_price:.2f}")
        with col3:
            price_change = df_prophet['y'].iloc[-1] - df_prophet['y'].iloc[-2]
            pct_change = (price_change / df_prophet['y'].iloc[-2]) * 100
            st.metric("📈 Daily Change", f"{pct_change:+.2f}%", delta=f"${price_change:+.2f}")
        with col4:
            st.metric("📅 Data Points", len(df_prophet))
        
        st.markdown("---")
        
        # Split data
        split_idx = int(len(df_prophet) * 0.8)
        train_data = df_prophet[:split_idx].copy()
        test_data = df_prophet[split_idx:].copy()
        
        # Train model
        with st.spinner("🤖 Training NeuralProphet model... (this takes ~30 seconds)"):
            progress_bar = st.progress(0)
            
            model = NeuralProphet(
                epochs=50,
                learning_rate=0.01,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                # changepoint_prior_scale=0.05,
            )
            
            # Train with progress
            metrics = model.fit(train_data, freq='D', progress=None)
            progress_bar.progress(100)
            
            st.success("✅ Model training complete!")
        
        # Make predictions on test set
        test_forecast = model.predict(test_data)
        
        # Calculate metrics
        y_true = test_data['y'].values
        y_pred = test_forecast['yhat1'].values
        
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
        
        # Future forecast
        with st.spinner(f"🔮 Forecasting next {forecast_days} days..."):
            future = model.make_future_dataframe(df_prophet, periods=forecast_days, n_historic_predictions=len(df_prophet))
            forecast = model.predict(future)
        
        # Plot
        st.markdown(f"### 📈 {ticker} Price Forecast")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            mode='lines',
            name='Historical',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Forecast
        future_dates = forecast['ds'][len(df_prophet):]
        future_pred = forecast['yhat1'][len(df_prophet):]
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_pred,
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        # Forecast confidence interval (if available)
        if 'yhat1_lower' in forecast.columns and 'yhat1_upper' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast['yhat1_upper'][len(df_prophet):],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast['yhat1_lower'][len(df_prophet):],
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
        forecast_df = pd.DataFrame({
            'Date': future_dates.dt.strftime('%Y-%m-%d'),
            'Predicted Price': future_pred.round(2),
            'Change from Last': (future_pred - df_prophet['y'].iloc[-1]).round(2),
            'Change %': ((future_pred - df_prophet['y'].iloc[-1]) / df_prophet['y'].iloc[-1] * 100).round(2)
        })
        
        st.dataframe(forecast_df, use_container_width=True)
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    # Initial state - show instructions
    st.info("👈 **Configure settings in the sidebar and click 'Train & Forecast' to begin!**")
    
    st.markdown("### 🎯 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Select Stock**
        - Enter ticker symbol (AAPL, GOOGL, TSLA, etc.)
        - Choose date range
        - Set forecast period
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Train Model**
        - Downloads historical data
        - Trains NeuralProphet model
        - Validates on test set
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ View Results**
        - Interactive forecast chart
        - Performance metrics
        - Download predictions
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Built by <strong>Alfred So</strong> | ML Engineer</p>
    <p>🎓 AWS ML • GCP ML • Azure AI • Databricks ML • NVIDIA AIIO</p>
</div>
""", unsafe_allow_html=True)
