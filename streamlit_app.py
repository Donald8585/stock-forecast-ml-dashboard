import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prophet import Prophet
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
st.markdown('<p class="subtitle">Prophet Time Series Forecasting System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
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
    st.markdown("### 🧠 Model Info")
    st.info("""
    **Prophet (Meta)**
    - Time series forecasting
    - Yearly + Weekly seasonality
    - Handles missing data
    - Production-ready
    """)
    
    st.markdown("### 🔗 Links")
    st.markdown("- [GitHub Repo](https://github.com/Donald8585/stock-forecast-ml-dashboard)")
    st.markdown("- [LinkedIn](https://linkedin.com/in/alfred-so)")

# Main content
if train_button:
    try:
        with st.spinner(f"📥 Downloading {ticker} data..."):
            # Try downloading real data
            use_demo = False
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, timeout=10)
                
                if df.empty or len(df) < 10:
                    st.warning(f"⚠️ Could not download {ticker} data. Using demo data instead...")
                    use_demo = True
            except Exception as e:
                st.warning(f"⚠️ Yahoo Finance error: {str(e)[:100]}. Using demo data instead...")
                use_demo = True
            
            # Use demo data if download failed
            if use_demo:
                # Generate realistic demo data
                date_range = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
                np.random.seed(42)
                
                # Simulate realistic stock price with trend, seasonality, and noise
                days = len(date_range)
                base_price = 100
                trend = np.linspace(0, 50, days)
                seasonality = 10 * np.sin(np.arange(days) * 2 * np.pi / 365)
                noise = np.random.normal(0, 3, days)
                
                close_prices = base_price + trend + seasonality + noise
                
                # Create demo DataFrame matching yfinance structure
                df = pd.DataFrame({
                    'Close': close_prices
                }, index=date_range)
                
                ticker_display = f"{ticker} (Demo)"
            else:
                ticker_display = ticker
            
            # Prepare data for Prophet
            df_clean = df.copy()
            
            # Handle MultiIndex columns if present
            if isinstance(df_clean.columns, pd.MultiIndex):
                df_clean.columns = df_clean.columns.get_level_values(0)
            
            # Create Prophet dataframe
            df_prophet = pd.DataFrame()
            df_prophet['ds'] = pd.to_datetime(df_clean.index)
            df_prophet['y'] = df_clean['Close'].values
            df_prophet = df_prophet.reset_index(drop=True)
            
            # Remove timezone if present
            if df_prophet['ds'].dt.tz is not None:
                df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
            
            st.success(f"✅ Loaded {len(df_prophet)} days of data")
            
            # Add mode badge
            if use_demo:
                st.markdown("""
                <div style='background: #FFF3CD; border: 2px solid #FFA500; border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
                    <h4 style='margin: 0; color: #856404;'>🎮 DEMO MODE</h4>
                    <p style='margin: 0.5rem 0 0 0; color: #856404;'>Using simulated stock data. Yahoo Finance rate limited - try again later for live data.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #D4EDDA; border: 2px solid #28A745; border-radius: 8px; padding: 1rem; margin: 1rem 0;'>
                    <h4 style='margin: 0; color: #155724;'>✅ LIVE DATA MODE</h4>
                    <p style='margin: 0.5rem 0 0 0; color: #155724;'>Real-time stock data from Yahoo Finance.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Stock", ticker_display.upper())
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
        with st.spinner("🤖 Training Prophet model... (this takes ~20 seconds)"):
            progress_bar = st.progress(0)
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Train
            model.fit(train_data)
            progress_bar.progress(100)
            
            st.success("✅ Model training complete!")
        
        # Make predictions on test set
        test_forecast = model.predict(test_data)
        
        # Calculate metrics
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
        
        # Future forecast
        with st.spinner(f"🔮 Forecasting next {forecast_days} days..."):
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
        
        # Plot
        st.markdown(f"### 📈 {ticker_display} Price Forecast")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            mode='lines',
            name='Historical',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Get only future predictions
        future_mask = forecast['ds'] > df_prophet['ds'].max()
        future_forecast_data = forecast[future_mask].head(forecast_days)
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_forecast_data['ds'],
            y=future_forecast_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_forecast_data['ds'],
            y=future_forecast_data['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_forecast_data['ds'],
            y=future_forecast_data['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title=f"{ticker_display} Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown("### 📋 Forecast Details")
        
        # Add mode indicator BEFORE table
        if use_demo:
            st.warning("⚠️ **Demo Mode** - Simulated predictions for demonstration purposes.")
        else:
            st.info(f"✅ **Live Data** - Real {ticker} stock predictions from Yahoo Finance.")
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_forecast_data['ds'].dt.strftime('%Y-%m-%d').values,
            'Predicted Price': future_forecast_data['yhat'].round(2).values,
            'Lower Bound': future_forecast_data['yhat_lower'].round(2).values,
            'Upper Bound': future_forecast_data['yhat_upper'].round(2).values,
            'Change from Last': (future_forecast_data['yhat'].values - df_prophet['y'].iloc[-1]).round(2),
            'Change %': ((future_forecast_data['yhat'].values - df_prophet['y'].iloc[-1]) / df_prophet['y'].iloc[-1] * 100).round(2)
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
        - Enter ticker symbol
        - Choose date range
        - Set forecast period
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Train Model**
        - Downloads historical data
        - Trains Prophet model
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
    st.markdown("### 💡 Suggested Stock Tickers")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.code("GOOGL\nAlphabet")
    with col2:
        st.code("MSFT\nMicrosoft")
    with col3:
        st.code("TSLA\nTesla")
    with col4:
        st.code("NVDA\nNVIDIA")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Built by <strong>Alfred So</strong> | ML Engineer</p>
    <p>🎓 AWS ML • GCP ML • Azure AI • Databricks ML • NVIDIA AIIO</p>
</div>
""", unsafe_allow_html=True)
