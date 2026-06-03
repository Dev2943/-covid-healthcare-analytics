FROM python:3.11-slim

# Prophet / cmdstanpy need a C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Cloud Run provides $PORT; Dash app exposes `server = app.server`
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 4 --timeout 120 dashboard.app:server
