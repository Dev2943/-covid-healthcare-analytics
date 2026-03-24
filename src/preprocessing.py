"""
preprocessing.py
----------------
Cleans, engineers features, and prepares data slices
for the three analytical lenses in this project:
  1. Geospatial spread
  2. Resource utilization (ICU / hospitalizations)
  3. ML forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def fill_time_series_gaps(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Forward-fill then backward-fill gaps within each country's time series."""
    df = df.sort_values(["location", "date"])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df.groupby("location")[col]
                .transform(lambda s: s.ffill().bfill())
            )
    return df


def add_rolling_averages(df: pd.DataFrame, cols: List[str], window: int = 7) -> pd.DataFrame:
    """Add 7-day rolling averages for specified columns (per country)."""
    df = df.sort_values(["location", "date"])
    for col in cols:
        if col in df.columns:
            df[f"{col}_7day_avg"] = (
                df.groupby("location")[col]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
    return df


def add_growth_rate(df: pd.DataFrame, col: str = "new_cases") -> pd.DataFrame:
    """Compute week-over-week growth rate for a given column."""
    df = df.sort_values(["location", "date"])
    df[f"{col}_growth_rate"] = (
        df.groupby("location")[col]
        .transform(lambda s: s.pct_change(periods=7).replace([np.inf, -np.inf], np.nan))
    )
    return df


# ── Geospatial Slice ─────────────────────────────────────────────────────────

def prepare_geospatial(df: pd.DataFrame, as_of_date: Optional[str] = None) -> pd.DataFrame:
    if as_of_date:
        df = df[df["date"] <= pd.to_datetime(as_of_date)]

    snapshot = (
        df.sort_values("date")
        .groupby(["iso_code", "location", "continent"])
        .last()
        .reset_index()
    )

    snapshot["case_fatality_rate"] = (
        snapshot["total_deaths"] / snapshot["total_cases"].replace(0, np.nan) * 100
    ).round(2)

    snapshot["hosp_per_million"] = snapshot.get("hosp_patients_per_million", np.nan)
    snapshot["icu_per_million"]  = snapshot.get("icu_patients_per_million",  np.nan)

    keep = [
        "iso_code", "location", "continent",
        "total_cases", "total_deaths",
        "case_fatality_rate",
        "hosp_per_million", "icu_per_million",
        "hospital_beds_per_thousand", "population",
    ]
    keep = [c for c in keep if c in snapshot.columns]
    return snapshot[keep].dropna(subset=["iso_code"])


# ── Resource Utilisation Slice ────────────────────────────────────────────────

def prepare_resource_utilization(
    df: pd.DataFrame,
    countries: List[str],
    start_date: str = "2020-01-01",
    end_date:   str = "2023-12-31",
) -> pd.DataFrame:
    mask = (
        df["location"].isin(countries) &
        (df["date"] >= pd.to_datetime(start_date)) &
        (df["date"] <= pd.to_datetime(end_date))
    )
    sub = df[mask].copy()

    resource_cols = ["icu_patients", "hosp_patients",
                     "icu_patients_per_million", "hosp_patients_per_million"]
    existing = [c for c in resource_cols if c in sub.columns]
    sub = fill_time_series_gaps(sub, existing)
    sub = add_rolling_averages(sub, existing, window=7)

    if "hosp_patients" in sub.columns and "hospital_beds_per_thousand" in sub.columns:
        sub["total_beds_est"] = sub["hospital_beds_per_thousand"] * sub["population"] / 1000
        sub["bed_occupancy_pct"] = (
            sub["hosp_patients"] / sub["total_beds_est"].replace(0, np.nan) * 100
        ).clip(upper=100).round(2)

    return sub.sort_values(["location", "date"])


# ── Forecasting Slice ─────────────────────────────────────────────────────────

def prepare_forecasting(
    df: pd.DataFrame,
    country: str,
    target_col: str = "new_cases_smoothed",
    start_date: str = "2020-03-01",
) -> pd.DataFrame:
    sub = df[
        (df["location"] == country) &
        (df["date"] >= pd.to_datetime(start_date))
    ][["date", target_col]].copy()

    sub = sub.dropna(subset=[target_col])
    sub[target_col] = sub[target_col].clip(lower=0)
    sub.columns = ["ds", "y"]
    sub = sub.sort_values("ds").reset_index(drop=True)
    return sub