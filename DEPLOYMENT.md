# 🚀 Deployment Guide

## Option 1: Streamlit Cloud (Recommended - FREE)

### Steps:
1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Stock forecast ML dashboard"
   git branch -M main
   git remote add origin https://github.com/Donald8585/stock-forecast-ml-dashboard.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Select your repository: `Donald8585/stock-forecast-ml-dashboard`
   - Set main file: `src/dashboard_fixed.py`
   - Advanced settings:
     ```
     Python version: 3.10
     ```
   - Click "Deploy"!

3. **⚠️ Note**: API won't work on Streamlit Cloud (frontend only)
   - Solution: Deploy API separately on Railway/Render (see below)
   - Or: Modify dashboard to run models locally (no API needed)

---

## Option 2: Railway (FREE tier - Full Stack)

### Deploy API + Dashboard Together

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Configure Services**
   - Railway will auto-detect your app
   - Set port to 8501 for dashboard
   - Get public URL

---

## Option 3: Render (FREE tier)

### Deploy as Web Service

1. **Go to [render.com](https://render.com/)**

2. **Create New Web Service**
   - Connect GitHub repo
   - Build Command: `pip install -r requirements.txt && python src/model_fixed.py`
   - Start Command: `streamlit run src/dashboard_fixed.py --server.port $PORT --server.address 0.0.0.0`

3. **Deploy API Separately**
   - Create another web service
   - Start Command: `uvicorn src.api_fixed:app --host 0.0.0.0 --port $PORT`

---

## Option 4: AWS (Production-Grade)

### Deploy on EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip
git clone https://github.com/Donald8585/stock-forecast-ml-dashboard.git
cd stock-forecast-ml-dashboard
pip3 install -r requirements.txt

# Train models
python3 src/model_fixed.py

# Run with PM2 (process manager)
sudo npm install -g pm2
pm2 start src/api_fixed.py --interpreter python3 --name api
pm2 start "streamlit run src/dashboard_fixed.py" --name dashboard

# Setup nginx reverse proxy (optional)
sudo apt install nginx
# Configure nginx to proxy port 80 -> 8501
```

---

## Option 5: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
```

---

## Option 6: Heroku

1. **Create Procfile**
   ```
   web: streamlit run src/dashboard_fixed.py --server.port $PORT
   ```

2. **Deploy**
   ```bash
   heroku login
   heroku create stock-forecast-app
   git push heroku main
   ```

---

## 🔗 Post-Deployment

### Update Dashboard API URL
If API and dashboard are deployed separately, update `dashboard_fixed.py`:

```python
API_URL = "https://your-api-url.com"  # Change from localhost

response = requests.post(
    f"{API_URL}/forecast",
    json={"periods": periods, "model": model}
)
```

### Add to Portfolio
- ✅ Add live link to LinkedIn
- ✅ Add to GitHub profile README
- ✅ Include in resume under "Projects"
- ✅ Share on Twitter/LinkedIn with demo video

---

## 🎯 Recommended for You (Alfred)

**For SF ML Engineer Jobs:**

1. **Deploy dashboard on Streamlit Cloud** (easiest, free)
2. **Deploy API on Railway** (free tier, professional)
3. **Add link to LinkedIn**: "Live Demo: [link]"
4. **Show in interviews**: Pull it up during technical discussions

This shows you can:
- ✅ Build production ML systems
- ✅ Create REST APIs
- ✅ Deploy to cloud
- ✅ Full-stack ML engineering
