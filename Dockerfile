# Use official slim Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies separately for caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY app_flask/ ./app_flask/
COPY app_streamlit/ ./app_streamlit/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose ports for Flask & Streamlit
EXPOSE 5001 8501

# Health check for Flask
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/ || exit 1

# Start both Flask (Gunicorn) and Streamlit
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5001 --workers 2 --timeout 120 app_flask.app:app & \
                  streamlit run app_streamlit/app.py --server.port=8501 --server.address=0.0.0.0"]

