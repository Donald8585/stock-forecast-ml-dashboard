FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Expose ports
EXPOSE 8000 8501

# Train models on startup (or mount pre-trained models)
RUN python src/model_fixed.py || echo "Models will be trained on first run"

# Start both API and dashboard
CMD uvicorn src.api_fixed:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard_fixed.py --server.port 8501 --server.address 0.0.0.0
