# 🦠 COVID-19 Healthcare Analytics Dashboard

> **A full end-to-end Business Analytics project** combining Python, ML forecasting, Monte Carlo risk simulation, and interactive dashboards to support healthcare resource planning decisions.

## 🚀 Live Demo

**[View Live Dashboard](https://covid-healthcare-analytics-909492874362.us-central1.run.app)**

> Deployed on **Google Cloud Run** with Docker — always on, auto-redeploys on every GitHub push.

---

## 📌 Project Overview

Healthcare systems worldwide faced unprecedented strain during the COVID-19 pandemic. This project analyses global outbreak data across **three analytical lenses** that directly map to real decisions hospital administrators and health authorities make:

| Lens | Business Question |
|------|------------------|
| 🌍 Geospatial Spread | Which regions are most severely impacted, and how has spread evolved over time? |
| 🏥 Resource Utilization | Are hospital and ICU capacities at risk of being overwhelmed? |
| 📈 ML Forecasting + Monte Carlo | How many cases should we expect — and how bad could it plausibly get? |

---

## 🎯 Business Impact

This type of analysis directly enables:
- **Surge planning** — activate overflow capacity *before* beds run out, not after
- **Staff rostering** — schedule additional ICU nurses 2–4 weeks ahead of forecast peaks
- **Supply chain** — pre-order ventilators, PPE, and medications based on predicted demand
- **Tail-risk capacity** — size standby capacity off the 95% worst-case scenario, not just the expected case
- **Executive reporting** — translate raw epidemiological data into board-level KPIs

---

## ✨ Key Features

- **3-tab interactive dashboard** — geospatial spread, resource utilization, and ML forecasting
- **Facebook Prophet forecasting** — 30/60/90-day horizon with automatic seasonality and changepoint detection
- **Monte Carlo scenario simulation** — 1,000 simulated outbreak trajectories with risk bands (the epidemiological analogue of Value-at-Risk)
- **"Case-at-Risk" KPI** — 95% worst-case peak, a single number for surge capacity planning
- **Surge detection logic** — alerts when forecast breaches 1.5× the 14-day baseline
- **Geospatial choropleths** — world map with date slider and metric selector
- **Resource utilization tracking** — ICU/hospital occupancy with 80% capacity alert lines
- **Always-on cloud deployment** — Dockerized on Google Cloud Run

---

## 🎲 Monte Carlo Scenario Simulation

A single Prophet forecast tells administrators the *expected* case load. But capacity planning is about **tail risk** — "how bad could it plausibly get?"

This project adds a Monte Carlo layer that simulates 1,000 possible future trajectories using a geometric random-walk calibrated to recent volatility (the same engine used in quantitative finance for option pricing), then reports decision-ready risk bands:

| Output | Planning Use |
|--------|-------------|
| Expected path (median) | Staffing & bed baseline |
| 50% scenario range | Likely operating envelope |
| 95% scenario range | Surge capacity to keep on standby |
| **Case-at-Risk (95%)** | Single worst-case peak KPI for executives |

This reframes epidemiological forecasting in the language of risk management — the same tail-risk thinking used in bank Value-at-Risk models, applied to hospital beds instead of dollars.

---

## 🗂 Project Structure

```
covid-healthcare-analytics/
│
├── src/
│   ├── data_loader.py       # Loads OWID dataset (baked into image at build)
│   ├── preprocessing.py     # Cleans data + prepares 3 analytical slices
│   ├── forecasting.py       # Facebook Prophet wrapper + surge detection
│   └── monte_carlo.py       # Monte Carlo scenario simulation + Case-at-Risk
│
├── dashboard/
│   └── app.py               # Plotly Dash multi-tab interactive dashboard
│
├── prepare_data.py          # Build-time data download + trim (keeps image small)
├── Dockerfile               # Container config for GCP Cloud Run (+ CmdStan build)
├── requirements.txt
└── README.md
```

---

## 📊 Data Source

**Our World in Data — COVID-19 Dataset**
- URL: https://github.com/owid/covid-19-data
- 417,000+ rows across 248 countries (2020–2024)
- Key variables: `new_cases_smoothed`, `new_deaths_smoothed`, `icu_patients`, `hosp_patients`, `hospital_beds_per_thousand`, `iso_code`

The dataset is downloaded and trimmed at **Docker build time** (`prepare_data.py`), so the running container starts instantly without a large runtime download.

---

## 🛠 Tech Stack

| Layer | Tool |
|-------|------|
| Data ingestion | `requests`, `pandas` |
| Data processing | `pandas`, `numpy` |
| ML forecasting | `Prophet` (Meta) + `cmdstanpy` |
| Risk simulation | `numpy` (Monte Carlo) |
| Visualisation | `Plotly`, `Dash`, `Dash Bootstrap Components` |
| Deployment | `Docker` + `Gunicorn` + `Google Cloud Run` |
| CI/CD | GitHub → Cloud Build (auto-deploy on push) |

---

## 🌍 Dashboard Features

### Tab 1 — Geospatial Spread
- World choropleth map with date slider
- Metric selector: total cases, deaths, CFR, ICU/hosp per million
- Top 15 countries bar chart
- CFR vs case volume scatter (bubble = population)

### Tab 2 — Resource Utilization
- ICU patients per million — multi-country time series (7-day avg)
- Hospital patients per million — multi-country time series
- Estimated bed occupancy % with 80% capacity alert line
- Monthly ICU heatmap (year × month)

### Tab 3 — ML Forecast + Monte Carlo
- Prophet 30/60/90-day forecast with 95% confidence interval
- **Monte Carlo fan chart** — 1,000 simulated trajectories with 50% and 95% bands
- **Expected Peak** and **Case-at-Risk (95%)** KPI cards
- Trend decomposition (trend + weekly + yearly components)
- Surge alert overlay when predictions breach threshold

---

## 🔧 Local Setup

```bash
# Clone the repository
git clone https://github.com/Dev2943/covid-healthcare-analytics.git
cd covid-healthcare-analytics

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python dashboard/app.py
```
Open **http://localhost:8050** in your browser.

> Note: Prophet requires a working CmdStan backend. The Dockerfile handles this automatically for cloud deployment via `python -m cmdstanpy.install_cmdstan`.

---

## 🐳 Docker / GCP Deployment

```bash
# Build the container (downloads + trims data, compiles CmdStan)
docker build -t covid-dashboard .

# Run locally in Docker
docker run -p 8080:8080 covid-dashboard
```

Deployed on **Google Cloud Run** with 2 GiB memory (Prophet is memory-intensive) and minimum 1 instance for always-on availability. Every push to `main` triggers an automatic rebuild via Cloud Build.

---

## 📈 ML Methodology — Prophet Forecasting

Facebook Prophet is well-suited for COVID data because it handles:
1. **Multiple seasonality** — weekly patterns (weekend reporting dips) and annual patterns
2. **Automatic changepoint detection** — structural breaks (lockdowns, vaccine rollouts, new variants)
3. **Uncertainty quantification** — 95% confidence intervals, not just point estimates
4. **Robustness to missing data** — common in healthcare reporting

Model evaluation uses **time-series cross-validation** (not a random split — critical for temporal data), reporting MAE, MAPE, RMSE, and coverage.

---

## 💡 Resume Bullet Points

**COVID-19 Healthcare Analytics Dashboard** | Python · Prophet · Monte Carlo · Plotly Dash · Docker · GCP

- Engineered a full end-to-end analytics pipeline ingesting a 417K-row global COVID dataset (Our World in Data) across 248 countries, with automated build-time data preparation, cleaning, and feature engineering
- Developed a Facebook Prophet time-series forecasting model with 30–90 day horizon and surge detection, quantifying 95% prediction intervals to support hospital resource pre-positioning
- Added a Monte Carlo scenario simulation (1,000 trajectories) producing a "Case-at-Risk" tail-risk KPI — reframing epidemiological forecasting in the language of Value-at-Risk
- Built a 3-tab interactive Plotly Dash dashboard and deployed it on Google Cloud Run with Docker and automated CI/CD from GitHub

---

## 🔮 Future Enhancements
- [ ] Add vaccination rate as an exogenous regressor in Prophet (`add_regressor`)
- [ ] Integrate excess mortality data for more robust CFR analysis
- [ ] County/state level drill-down for the United States (CDC dataset)
- [ ] Email/Slack alert when surge threshold is breached
- [ ] Multivariate Monte Carlo (correlated case + hospitalization paths)

---

## 👨‍💻 Author

**Dev Golakiya** — MS Business Analytics, UMass Amherst
- 📧 devgolakiya07@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/devgolakiya)
- 🐙 [GitHub](https://github.com/Dev2943)

---

## 📄 License
MIT — free to use, fork, and adapt for your own portfolio.

## 🙏 Data Attribution
Hannah Ritchie, Edouard Mathieu, et al. (2020) — *Coronavirus Pandemic (COVID-19)*. Published online at OurWorldInData.org.

---

*Deployed on Google Cloud Run | Auto-deploys from GitHub | Last updated June 2026*
