FROM python:3.11-slim

# Prophet / cmdstanpy need a C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the COVID dataset at BUILD time so the server starts instantly
# (no large download blocking startup → no Cloud Run timeout)
RUN mkdir -p /app/data && \
    curl -sSL "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv" \
    -o /app/data/owid_covid_data.csv && \
    echo "Dataset baked into image: $(du -h /app/data/owid_covid_data.csv)"

EXPOSE 8080

# --timeout 0 disables the worker timeout (Prophet forecasts can be slow);
# --preload loads the app once before forking so data parse happens at boot
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 dashboard.app:server
