# 🦠 COVID-19 Healthcare Analytics Dashboard

> **A full end-to-end Business Analytics project** combining Python, ML forecasting, and interactive dashboards to support healthcare resource planning decisions.

---

## 📌 Project Overview

Healthcare systems worldwide faced unprecedented strain during the COVID-19 pandemic. This project analyses global outbreak data across **three analytical lenses** that directly map to real decisions hospital administrators and health authorities make:

| Lens | Business Question |
|------|------------------|
| 🌍 Geospatial Spread | Which regions are most severely impacted, and how has spread evolved over time? |
| 🏥 Resource Utilization | Are hospital and ICU capacities at risk of being overwhelmed? |
| 📈 ML Forecasting | How many cases / patients should we expect in the next 30–90 days? |

---

## 🎯 Business Impact

This type of analysis directly enables:
- **Surge planning** — activate overflow capacity *before* beds run out, not after
- **Staff rostering** — schedule additional ICU nurses 2–4 weeks ahead of forecast peaks
- **Supply chain** — pre-order ventilators, PPE, and medications based on predicted demand
- **Executive reporting** — translate raw epidemiological data into board-level KPIs

> **Example value statement for your resume / portfolio:**
> *"Built a 30-day ML forecasting model (Prophet) with surge detection logic that could give hospital administrators a 2–4 week planning window ahead of ICU surges, potentially preventing capacity crises."*

---

## 🗂 Project Structure

```
covid-healthcare-analytics/
│
├── src/
│   ├── data_loader.py       # Downloads & caches Our World in Data dataset
│   ├── preprocessing.py     # Cleans data + prepares 3 analytical slices
│   └── forecasting.py       # Facebook Prophet wrapper + surge detection
│
├── dashboard/
│   └── app.py               # Plotly Dash multi-tab interactive dashboard
│
├── notebooks/               # (Optional) Exploratory analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_geospatial_analysis.ipynb
│   ├── 03_resource_utilization.ipynb
│   └── 04_forecasting_deep_dive.ipynb
│
├── data/                    # Auto-created on first run (gitignored)
│   └── owid_covid_data.csv
│
├── requirements.txt
└── README.md
```

---

## 📊 Data Source

**Our World in Data — COVID-19 Dataset**
- URL: https://github.com/owid/covid-19-data
- Updated daily; 67+ variables per country
- Key variables used:
  - `new_cases_smoothed`, `new_deaths_smoothed` — epidemiological trends
  - `icu_patients`, `hosp_patients` (per million) — resource utilization
  - `hospital_beds_per_thousand` — capacity baseline
  - `iso_code` — for choropleth mapping

No manual download required — `data_loader.py` fetches it automatically.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/covid-healthcare-analytics.git
cd covid-healthcare-analytics
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ Prophet requires `pystan`. If you hit install issues on Windows, use:
> `conda install -c conda-forge prophet`

### 4. Run the dashboard
```bash
python dashboard/app.py
```
Open **http://localhost:8050** in your browser.

---

## 🛠 Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Data Ingestion | `requests`, `pandas` | Automated download + caching |
| Data Processing | `pandas`, `numpy` | Cleaning, rolling averages, feature engineering |
| ML Forecasting | `Prophet` (Meta) | Handles seasonality, holidays, trend changepoints automatically |
| Visualisation | `Plotly`, `Dash` | Interactive charts; geospatial choropleths |
| Dashboard | `Dash Bootstrap Components` | Professional dark-themed UI |
| Deployment | `Gunicorn` / Render.com | `server = app.server` exposes WSGI endpoint |

---

## 📈 ML Methodology — Prophet Forecasting

### Why Prophet?
Facebook Prophet is well-suited for COVID data because:
1. **Multiple seasonality** — captures weekly patterns (weekend reporting dips) and annual patterns simultaneously
2. **Automatic changepoint detection** — identifies structural breaks (lockdowns, vaccine rollouts, new variants) without manual labelling
3. **Uncertainty quantification** — produces 95% confidence intervals so decision-makers understand the risk range, not just a point estimate
4. **Robustness to missing data** — common in healthcare reporting

### Surge Detection Logic
A "surge alert" fires when the model predicts that cases will exceed **1.5× the 14-day rolling baseline**. This mirrors early-warning systems used by real health authorities.

### Model Evaluation
```python
from src.forecasting import evaluate_model
metrics = evaluate_model(df_prophet)
# Returns: {'mae': ..., 'mape': ..., 'rmse': ..., 'coverage': ...}
```
Metrics are computed via time-series cross-validation (not random split — critical for temporal data).

---

## 📉 Feature Engineering

| Feature | Description | Business Use |
|---------|-------------|-------------|
| `7-day rolling average` | Smooths weekend reporting dips | More reliable trend signal |
| `week-over-week growth rate` | % change vs 7 days ago | Early surge indicator |
| `bed_occupancy_pct` | Hosp patients ÷ estimated total beds | Capacity stress KPI |
| `case_fatality_rate` | Deaths ÷ Cases | Cross-country severity comparison |

---

## 🌍 Dashboard Features

### Tab 1 — Geospatial Spread
- World choropleth map with date slider
- Metric selector: total cases, deaths, CFR, ICU/hosp per million
- Top 15 countries bar chart
- CFR vs Case Volume scatter (bubble = population)

### Tab 2 — Resource Utilization
- ICU patients per million — multi-country time series (7-day avg)
- Hospital patients per million — multi-country time series
- Estimated bed occupancy % with 80% capacity alert line
- Monthly ICU heatmap (year × month) for pattern identification

### Tab 3 — ML Forecast
- Prophet 30/60/90-day forecast with 95% confidence interval
- Surge alert overlay (red shading) when predictions breach threshold
- Trend decomposition (trend + weekly + yearly components)
- Executive business impact summary card

---

## 💡 How to Use This on Your Resume

### Project Title
**COVID-19 Healthcare Analytics Dashboard** | Python · Prophet · Plotly Dash · Pandas

### Bullet Points
- Engineered a full end-to-end analytics pipeline ingesting 67-variable global COVID dataset (Our World in Data) across 200+ countries, building automated data refresh, cleaning, and feature engineering modules in Python
- Developed a Facebook Prophet time-series forecasting model with 30–90 day horizon and surge detection logic, quantifying 95% prediction intervals to support hospital resource pre-positioning decisions
- Built a 3-tab interactive Plotly Dash dashboard covering geospatial spread (choropleth), ICU/bed utilisation with 80%-capacity alerts, and ML forecast decomposition

---

## 🔮 Future Enhancements
- [ ] Add vaccination rate as an exogenous regressor in Prophet (`add_regressor`)
- [ ] Integrate excess mortality data for more robust CFR analysis
- [ ] Add county/state level drill-down for the United States (CDC dataset)
- [ ] Deploy to Render.com or Heroku for live public demo link
- [ ] Add email/Slack alert when surge threshold is breached

---

## 📄 License
MIT — free to use, fork, and adapt for your own portfolio.

---

## 🙏 Data Attribution
Hannah Ritchie, Edouard Mathieu, et al. (2020) — *Coronavirus Pandemic (COVID-19)*. Published online at OurWorldInData.org.
