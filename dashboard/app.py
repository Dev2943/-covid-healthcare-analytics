"""
app.py  —  COVID-19 Healthcare Analytics Dashboard
---------------------------------------------------
Full-stack Plotly Dash application covering:
  Tab 1 — Geospatial Spread         (choropleth + time slider)
  Tab 2 — Resource Utilization      (ICU / hosp trends + capacity stress)
  Tab 3 — ML Forecasting            (Prophet 30-day case forecast + surge alerts)

Run:
    pip install -r requirements.txt
    python dashboard/app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from src.data_loader import load_and_filter
from src.preprocessing import (
    prepare_geospatial,
    prepare_resource_utilization,
    prepare_forecasting,
)
from src.forecasting import run_forecast_pipeline

# ── Colour palette ─────────────────────────────────────────────────────────
BRAND_BLUE    = "#1a73e8"
BRAND_RED     = "#d93025"
BRAND_ORANGE  = "#f4a261"
BRAND_GREEN   = "#2a9d8f"
BRAND_PURPLE  = "#6a4c93"
BG_DARK       = "#060a0f"
CARD_BG       = "#0d1117"
TEXT_LIGHT    = "#e8eaf6"
GRID_COLOR    = "#1a1f2e"
VIRUS_COLOR   = "#1a4a2e"

CHART_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_LIGHT, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

DEFAULT_COUNTRIES = ["United States", "United Kingdom", "Germany",
                     "India", "Brazil", "France"]

# ── Virus SVG background ────────────────────────────────────────────────────
# A single coronavirus SVG shape (circle + spikes)
def virus_svg(cx, cy, r, opacity):
    spikes = ""
    num_spikes = 12
    for i in range(num_spikes):
        angle = (360 / num_spikes) * i
        import math
        rad = math.radians(angle)
        x1 = cx + r * math.cos(rad)
        y1 = cy + r * math.sin(rad)
        x2 = cx + (r + r * 0.55) * math.cos(rad)
        y2 = cy + (r + r * 0.55) * math.sin(rad)
        ball_r = r * 0.13
        spikes += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#00ff88" stroke-width="{r*0.07:.1f}" opacity="{opacity}"/>'
        spikes += f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="{ball_r:.1f}" fill="#00ff88" opacity="{opacity*0.9:.2f}"/>'
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#00cc66" '
        f'stroke-width="{r*0.07:.1f}" opacity="{opacity}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r*0.6:.1f}" fill="#051a0f" opacity="{opacity*0.5:.2f}"/>'
        + spikes
    )

# Build background SVG with many viruses scattered around
virus_positions = [
    (80,  80,  55, 0.07),
    (300, 150, 35, 0.05),
    (700, 50,  65, 0.06),
    (1100, 200, 45, 0.05),
    (1350, 80,  70, 0.07),
    (150, 400, 40, 0.04),
    (500, 350, 60, 0.06),
    (900, 300, 30, 0.04),
    (1250, 400, 55, 0.05),
    (50,  650, 50, 0.05),
    (350, 600, 75, 0.07),
    (750, 550, 35, 0.04),
    (1050, 600, 65, 0.06),
    (1400, 550, 40, 0.05),
    (200, 850, 45, 0.05),
    (600, 800, 55, 0.06),
    (1000, 800, 40, 0.04),
    (1300, 750, 60, 0.06),
]

all_viruses = "".join(virus_svg(x, y, r, op) for x, y, r, op in virus_positions)

background_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%"
     style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <g filter="url(#glow)">
    {all_viruses}
  </g>
</svg>'''

# ── Data loading (cached at module level) ───────────────────────────────────
print("Loading data …")
raw_df = load_and_filter()
all_countries = sorted(raw_df["location"].dropna().unique().tolist())
date_min = raw_df["date"].min().strftime("%Y-%m-%d")
date_max = raw_df["date"].max().strftime("%Y-%m-%d")
print(f"Data loaded: {len(raw_df):,} rows | {raw_df['location'].nunique()} countries")

# ── App initialisation ──────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    title="COVID-19 Healthcare Analytics",
    suppress_callback_exceptions=True,
)
server = app.server

# ── Reusable components ─────────────────────────────────────────────────────

def stat_card(title: str, value: str, icon: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"fa {icon} fa-2x", style={"color": color}),
                html.Div([
                    html.P(title, className="text-muted mb-0", style={"fontSize": "0.8rem"}),
                    html.H5(value, style={"color": color, "fontWeight": "700"}),
                ], className="ms-3"),
            ], className="d-flex align-items-center"),
        ]),
        style={
            "backgroundColor": "rgba(13,17,23,0.85)",
            "border": f"1px solid {color}55",
            "backdropFilter": "blur(8px)",
        },
    )


# ── Layout ──────────────────────────────────────────────────────────────────

app.layout = html.Div([

    # ── Virus background layer
    dcc.Markdown(
        background_svg,
        dangerously_allow_html=True,
        style={"position": "fixed", "top": 0, "left": 0,
               "width": "100%", "height": "100%",
               "zIndex": 0, "pointerEvents": "none"},
    ),

    # ── Main content (above the background)
    dbc.Container([

        # ── Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("🦠", style={"fontSize": "2.5rem", "marginRight": "12px"}),
                    html.Span("COVID-19 Healthcare Analytics",
                              style={
                                  "color": TEXT_LIGHT,
                                  "fontWeight": "800",
                                  "fontSize": "1.8rem",
                                  "textShadow": "0 0 20px #00ff8844",
                              }),
                ], style={"display": "flex", "alignItems": "center"}),
                html.P(
                    f"Data: Our World in Data  ·  Updated through {date_max}  ·  "
                    "Built for Healthcare Business Analytics Portfolio",
                    className="text-muted", style={"fontSize": "0.85rem"},
                ),
            ])
        ], className="mt-3 mb-2"),

        # ── KPI strip
        dbc.Row(id="kpi-row", className="mb-3"),

        # ── Main tabs
        dbc.Tabs([

            # ── TAB 1: Geospatial ──────────────────────────────────────────
            dbc.Tab(label="🌍 Geospatial Spread", tab_id="tab-geo", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Metric", className="text-muted mt-3"),
                        dcc.Dropdown(
                            id="geo-metric",
                            options=[
                                {"label": "Total Cases",              "value": "total_cases"},
                                {"label": "Total Deaths",             "value": "total_deaths"},
                                {"label": "Case Fatality Rate (%)",   "value": "case_fatality_rate"},
                                {"label": "ICU Patients per Million", "value": "icu_per_million"},
                                {"label": "Hosp. Patients per Million","value": "hosp_per_million"},
                            ],
                            value="total_cases",
                            clearable=False,
                            style={"backgroundColor": CARD_BG, "color": "#000"},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("As of Date", className="text-muted mt-3"),
                        dcc.DatePickerSingle(
                            id="geo-date",
                            min_date_allowed=date_min,
                            max_date_allowed=date_max,
                            date=date_max,
                            display_format="YYYY-MM-DD",
                            style={"backgroundColor": "#0d1117", "border": "1px solid #1a73e8", "borderRadius": "6px"},
                            className="dark-date-picker",
                        ),
                    ], width=4),
                ]),
                dcc.Graph(id="choropleth-map", style={"height": "520px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="top-countries-bar", style={"height": "350px"})),
                    dbc.Col(dcc.Graph(id="cfr-scatter",       style={"height": "350px"})),
                ]),
            ]),

            # ── TAB 2: Resource Utilization ────────────────────────────────
            dbc.Tab(label="🏥 Resource Utilization", tab_id="tab-res", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Countries", className="text-muted mt-3"),
                        dcc.Dropdown(
                            id="res-countries",
                            options=[{"label": c, "value": c} for c in all_countries],
                            value=["United States", "United Kingdom", "Germany"],
                            multi=True,
                            style={"backgroundColor": CARD_BG, "color": "#000"},
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Date Range", className="text-muted mt-3"),
                        dcc.DatePickerRange(
                            id="res-date-range",
                            min_date_allowed=date_min,
                            max_date_allowed=date_max,
                            start_date="2020-03-01",
                            end_date=date_max,
                            display_format="YYYY-MM-DD",
                            style={"backgroundColor": "#0d1117", "border": "1px solid #1a73e8", "borderRadius": "6px"},
                            className="dark-date-picker",
                        ),
                    ], width=6),
                ]),

                # Info box about data availability
                dbc.Alert([
                    html.I(className="fa fa-info-circle me-2"),
                    "💡 ICU & hospital data is only available for countries that reported it "
                    "to OWID (mainly Europe & USA). Try: Italy, France, Spain, Netherlands, "
                    "United Kingdom, United States, Canada, Australia.",
                ], color="info", className="mt-2 mb-2",
                   style={"fontSize": "0.85rem", "backgroundColor": "#0d2137", "border": "1px solid #1a73e8"}),

                dbc.Row([
                    dbc.Col(dcc.Graph(id="icu-trend",  style={"height": "350px"})),
                    dbc.Col(dcc.Graph(id="hosp-trend", style={"height": "350px"})),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="bed-occupancy",   style={"height": "350px"})),
                    dbc.Col(dcc.Graph(id="resource-heatmap",style={"height": "350px"})),
                ]),
            ]),

            # ── TAB 3: Forecasting ─────────────────────────────────────────
            dbc.Tab(label="📈 ML Forecast", tab_id="tab-fc", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Country", className="text-muted mt-3"),
                        dcc.Dropdown(
                            id="fc-country",
                            options=[{"label": c, "value": c} for c in all_countries],
                            value="United States",
                            clearable=False,
                            style={"backgroundColor": CARD_BG, "color": "#000"},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Target Variable", className="text-muted mt-3"),
                        dcc.Dropdown(
                            id="fc-target",
                            options=[
                                {"label": "Daily New Cases (smoothed)",  "value": "new_cases_smoothed"},
                                {"label": "Daily New Deaths (smoothed)", "value": "new_deaths_smoothed"},
                                {"label": "ICU Patients",                "value": "icu_patients"},
                                {"label": "Hospitalised Patients",       "value": "hosp_patients"},
                            ],
                            value="new_cases_smoothed",
                            clearable=False,
                            style={"backgroundColor": CARD_BG, "color": "#000"},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Forecast Horizon (days)", className="text-muted mt-3"),
                        dcc.Slider(id="fc-horizon", min=7, max=90, step=7, value=30,
                                   marks={7: "7d", 30: "30d", 60: "60d", 90: "90d"}),
                    ], width=4),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col(
                        dbc.Button("▶ Run Forecast", id="fc-run-btn",
                                   color="primary", className="mt-2"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Spinner(html.Div(id="fc-status"), color="primary"),
                        width="auto", className="mt-2",
                    ),
                ]),

                dcc.Graph(id="forecast-chart", style={"height": "450px"}),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="forecast-components", style={"height": "350px"})),
                    dbc.Col([
                        html.H6("📊 Business Impact Summary",
                                style={"color": TEXT_LIGHT, "marginTop": "20px"}),
                        html.Div(id="forecast-summary-table"),
                    ]),
                ]),
            ]),

        ], id="main-tabs", active_tab="tab-geo"),

    ], fluid=True, style={
        "backgroundColor": "transparent",
        "minHeight": "100vh",
        "padding": "0 20px",
        "position": "relative",
        "zIndex": 1,
    }),

], style={"backgroundColor": BG_DARK, "minHeight": "100vh"})


# ── Callbacks ────────────────────────────────────────────────────────────────

# KPI strip
@app.callback(Output("kpi-row", "children"), Input("main-tabs", "active_tab"))
def update_kpis(_):
    snap = prepare_geospatial(raw_df)
    total_cases  = f"{snap['total_cases'].sum() / 1e6:.1f}M"
    total_deaths = f"{snap['total_deaths'].sum() / 1e6:.2f}M"
    cfr          = f"{(snap['total_deaths'].sum() / snap['total_cases'].sum() * 100):.2f}%"
    countries    = str(snap["location"].nunique())

    cards = [
        stat_card("Total Cases",      total_cases,  "fa-virus",        BRAND_BLUE),
        stat_card("Total Deaths",     total_deaths, "fa-heart-broken", BRAND_RED),
        stat_card("Global CFR",       cfr,          "fa-percent",      BRAND_ORANGE),
        stat_card("Countries Tracked",countries,    "fa-globe",        BRAND_GREEN),
    ]
    return [dbc.Col(c, xs=6, md=3) for c in cards]


# Choropleth
@app.callback(
    Output("choropleth-map",    "figure"),
    Output("top-countries-bar", "figure"),
    Output("cfr-scatter",       "figure"),
    Input("geo-metric", "value"),
    Input("geo-date",   "date"),
)
def update_geo(metric, as_of_date):
    geo = prepare_geospatial(raw_df, as_of_date=as_of_date)

    label_map = {
        "total_cases":        "Total Cases",
        "total_deaths":       "Total Deaths",
        "case_fatality_rate": "Case Fatality Rate (%)",
        "icu_per_million":    "ICU Patients per Million",
        "hosp_per_million":   "Hosp. Patients per Million",
    }
    label = label_map.get(metric, metric)

    choropleth = px.choropleth(
        geo, locations="iso_code", color=metric,
        hover_name="location",
        hover_data={"total_cases": ":,.0f", "total_deaths": ":,.0f",
                    "case_fatality_rate": ":.2f"},
        color_continuous_scale="Reds",
        labels={metric: label},
        title=f"Global {label} — as of {as_of_date}",
    )
    choropleth.update_layout(**CHART_LAYOUT, geo=dict(bgcolor=CARD_BG, showframe=False))

    top15 = geo.nlargest(15, metric)
    bar = px.bar(
        top15, x=metric, y="location", orientation="h",
        color=metric, color_continuous_scale="Blues",
        title=f"Top 15 Countries — {label}",
        labels={metric: label, "location": ""},
    )
    bar.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_LIGHT, family='Inter, sans-serif'),
        margin=dict(l=200, r=40, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(size=12), ticksuffix='  '),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        title=f'Top 15 Countries — {label}',
    )

    scatter_df = geo.dropna(subset=["case_fatality_rate", "total_cases", "continent"]).copy()
    scatter_df["continent"] = scatter_df["continent"].astype(str)
    continent_colors = {
        "Africa": "#1a73e8", "Asia": "#d93025", "Europe": "#2a9d8f",
        "North America": "#f4a261", "South America": "#6a4c93",
        "Oceania": "#e9c46a", "Antarctica": "#aaaaaa",
    }
    scatter = go.Figure()
    for continent in sorted(scatter_df["continent"].unique()):
        sub = scatter_df[scatter_df["continent"] == continent]
        scatter.add_trace(go.Scatter(
            x=sub["total_cases"], y=sub["case_fatality_rate"],
            mode="markers",
            name=continent,
            text=sub["location"],
            marker=dict(
                size=sub["population"].fillna(1e6).apply(lambda x: max(5, min(40, (x/1e6)**0.5 * 3))),
                color=continent_colors.get(continent, "#888888"),
                opacity=0.7,
            ),
            hovertemplate="<b>%{text}</b><br>Cases: %{x:,.0f}<br>CFR: %{y:.2f}%<extra></extra>",
        ))
    scatter.update_xaxes(type="log", title="Total Cases (log)")
    scatter.update_yaxes(title="CFR (%)")
    scatter.update_layout(**CHART_LAYOUT, title="Case Fatality Rate vs Total Cases")

    return choropleth, bar, scatter


# Resource utilization
@app.callback(
    Output("icu-trend",        "figure"),
    Output("hosp-trend",       "figure"),
    Output("bed-occupancy",    "figure"),
    Output("resource-heatmap", "figure"),
    Input("res-countries",   "value"),
    Input("res-date-range",  "start_date"),
    Input("res-date-range",  "end_date"),
)
def update_resource(countries, start_date, end_date):
    if not countries:
        empty = go.Figure().update_layout(**CHART_LAYOUT)
        return empty, empty, empty, empty

    res = prepare_resource_utilization(raw_df, countries, start_date, end_date)

    colors = [BRAND_BLUE, BRAND_RED, BRAND_GREEN, BRAND_ORANGE,
              BRAND_PURPLE, "#e9c46a", "#f4a261"]

    def line_chart(col_7d, title, ylab):
        fig = go.Figure()
        has_data = False
        for i, country in enumerate(countries):
            sub = res[res["location"] == country]
            if col_7d in sub.columns:
                data = sub[sub[col_7d].notna()]
                if len(data) > 0:
                    has_data = True
                    fig.add_trace(go.Scatter(
                        x=data["date"], y=data[col_7d],
                        name=country, mode="lines",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
        if not has_data:
            fig.add_annotation(
                text="⚠️ No data available for selected countries.<br>"
                     "Try: Italy, France, Spain, Netherlands, UK, or USA.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color=BRAND_ORANGE),
                align="center",
            )
        fig.update_layout(**CHART_LAYOUT, title=title, yaxis_title=ylab, xaxis_title="")
        return fig

    icu_fig  = line_chart("icu_patients_per_million_7day_avg",
                          "ICU Patients per Million (7-day avg)", "Per Million")
    hosp_fig = line_chart("hosp_patients_per_million_7day_avg",
                          "Hospitalised Patients per Million (7-day avg)", "Per Million")

    # Bed occupancy
    occ_fig = go.Figure()
    has_occ = False
    for i, country in enumerate(countries):
        sub = res[res["location"] == country]
        if "bed_occupancy_pct" in sub.columns:
            data = sub[sub["bed_occupancy_pct"].notna()]
            if len(data) > 0:
                has_occ = True
                occ_fig.add_trace(go.Scatter(
                    x=data["date"], y=data["bed_occupancy_pct"],
                    name=country, mode="lines", fill="tozeroy",
                    line=dict(color=colors[i % len(colors)], width=1.5),
                ))
    if not has_occ:
        occ_fig.add_annotation(
            text="⚠️ No bed occupancy data for selected countries.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=BRAND_ORANGE), align="center",
        )
    else:
        occ_fig.add_hline(y=80, line_dash="dash", line_color=BRAND_RED,
                          annotation_text="⚠ 80% capacity alert")
    occ_fig.update_layout(**CHART_LAYOUT, title="Estimated Hospital Bed Occupancy (%)",
                          yaxis_title="%", xaxis_title="")

    # Monthly heatmap
    heatmap_country = countries[0]
    sub_hm = res[res["location"] == heatmap_country].copy()
    sub_hm["month"] = sub_hm["date"].dt.to_period("M").astype(str)
    if "icu_patients_per_million" in sub_hm.columns and sub_hm["icu_patients_per_million"].notna().any():
        monthly = sub_hm.groupby("month")["icu_patients_per_million"].mean().reset_index()
        monthly["year"]        = monthly["month"].str[:4]
        monthly["month_short"] = pd.to_datetime(monthly["month"]).dt.strftime("%b")
        pivot = monthly.pivot_table(index="year", columns="month_short",
                                    values="icu_patients_per_million")
        hm = px.imshow(
            pivot, color_continuous_scale="YlOrRd",
            title=f"Monthly Avg ICU Patients/Million — {heatmap_country}",
            aspect="auto",
        )
        hm.update_layout(**CHART_LAYOUT)
    else:
        hm = go.Figure()
        hm.add_annotation(
            text=f"⚠️ No ICU heatmap data for {heatmap_country}.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=BRAND_ORANGE), align="center",
        )
        hm.update_layout(**CHART_LAYOUT, title=f"ICU Data — {heatmap_country}")

    return icu_fig, hosp_fig, occ_fig, hm


# Forecasting
@app.callback(
    Output("forecast-chart",       "figure"),
    Output("forecast-components",  "figure"),
    Output("forecast-summary-table","children"),
    Output("fc-status",            "children"),
    Input("fc-run-btn",  "n_clicks"),
    Input("fc-country",  "value"),
    Input("fc-target",   "value"),
    Input("fc-horizon",  "value"),
)
def run_forecast(n_clicks, country, target_col, horizon):
    import traceback

    try:
        df_p = prepare_forecasting(raw_df, country=country,
                                   target_col=target_col, start_date="2020-03-01")

        if len(df_p) < 60:
            msg = f"⚠ Not enough data for {country} ({target_col}). Try a different target."
            empty = go.Figure().update_layout(**CHART_LAYOUT)
            return empty, empty, msg, msg

        results  = run_forecast_pipeline(df_p, forecast_horizon_days=horizon, run_eval=False)
        forecast = results["forecast"]
        future   = results["future"]
        surge    = results["surge"]

        label_map = {
            "new_cases_smoothed":  "Daily New Cases (7-day avg)",
            "new_deaths_smoothed": "Daily New Deaths (7-day avg)",
            "icu_patients":        "ICU Patients",
            "hosp_patients":       "Hospitalised Patients",
        }
        label = label_map.get(target_col, target_col)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_p["ds"], y=df_p["y"],
            name="Actual", mode="lines",
            line=dict(color=BRAND_BLUE, width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=future["ds"], y=future["yhat"],
            name=f"{horizon}-day Forecast", mode="lines",
            line=dict(color=BRAND_ORANGE, width=2.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([future["ds"], future["ds"][::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(244, 162, 97, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval",
            showlegend=True,
        ))

        surge_days = surge[surge["is_surge"] & (surge["ds"] > df_p["ds"].max())]
        if not surge_days.empty:
            fig.add_vrect(
                x0=surge_days["ds"].min(), x1=surge_days["ds"].max(),
                fillcolor=BRAND_RED, opacity=0.15,
                annotation_text="⚠ Surge Alert", annotation_position="top left",
                line_width=0,
            )

        fig.update_layout(
            **CHART_LAYOUT,
            title=f"Prophet Forecast — {label} in {country} (+{horizon} days)",
            xaxis_title="", yaxis_title=label,
        )

        comp_cols = [c for c in ["trend", "weekly", "yearly"] if c in forecast.columns]
        comp_fig  = make_subplots(rows=max(len(comp_cols), 1), cols=1,
                                   subplot_titles=[c.capitalize() for c in comp_cols] or [""])
        comp_colors = [BRAND_BLUE, BRAND_GREEN, BRAND_PURPLE]
        for i, col in enumerate(comp_cols, 1):
            comp_fig.add_trace(
                go.Scatter(x=forecast["ds"], y=forecast[col],
                           line=dict(color=comp_colors[i - 1], width=2), name=col),
                row=i, col=1,
            )
        comp_fig.update_layout(**CHART_LAYOUT, title="Forecast Decomposition",
                               showlegend=False, height=350)

        peak_date = future.loc[future["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d")
        peak_val  = f"{future['yhat'].max():,.0f}"
        end_val   = f"{future.iloc[-1]['yhat']:,.0f}"
        trend_dir = "📈 Rising" if future.iloc[-1]["yhat"] > future.iloc[0]["yhat"] else "📉 Falling"
        surge_ct  = len(surge_days)

        table = dbc.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([
                html.Tr([html.Td("Forecast Target"), html.Td(label)]),
                html.Tr([html.Td("Country"),          html.Td(country)]),
                html.Tr([html.Td("Horizon"),           html.Td(f"{horizon} days")]),
                html.Tr([html.Td("Peak Predicted"),    html.Td(f"{peak_val} on {peak_date}")]),
                html.Tr([html.Td("End-of-Horizon"),    html.Td(end_val)]),
                html.Tr([html.Td("Trend Direction"),   html.Td(trend_dir)]),
                html.Tr([html.Td("Surge Alert Days"),  html.Td(
                    html.Span(f"{surge_ct} days",
                              style={"color": BRAND_RED if surge_ct > 0 else BRAND_GREEN})
                )]),
            ]),
        ], bordered=True, hover=True, responsive=True, size="sm",
           style={"color": TEXT_LIGHT, "backgroundColor": CARD_BG})

        return fig, comp_fig, table, f"✅ Forecast complete for {country}"

    except Exception as e:
        err = traceback.format_exc()
        print("FORECAST ERROR:", err)
        empty = go.Figure().update_layout(**CHART_LAYOUT)
        error_msg = html.Div([
            html.P(f"❌ Error: {str(e)}", style={"color": "red"}),
            html.Pre(err, style={"color": "orange", "fontSize": "0.7rem"}),
        ])
        return empty, empty, error_msg, f"❌ Error: {str(e)}"


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)