FROM python:3.11-slim

# Prophet / cmdstanpy need a C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build the CmdStan backend that Prophet needs (fixes 'no attribute stan_backend')
RUN python -m cmdstanpy.install_cmdstan --verbose

COPY . .

# Download + trim the COVID dataset at BUILD time → small file, instant startup
RUN python prepare_data.py

EXPOSE 8080

# --timeout 0 disables worker timeout (Prophet forecasts can be slow)
# --preload imports the app once in the master before forking (loads data at boot)
CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 --preload dashboard.app:server
