"""
forecasting.py
--------------
Wraps Facebook Prophet for COVID case / hospitalisation forecasting.

Business framing
----------------
A 30-day rolling forecast gives hospital administrators a planning
window to pre-position staff, beds, and ventilators before a surge hits.
We quantify forecast uncertainty so decision-makers understand risk ranges.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet not installed. Run: pip install prophet")
    PROPHET_AVAILABLE = False


# ── Core Forecasting ─────────────────────────────────────────────────────────

def build_model(
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
) -> "Prophet":
    """
    Instantiate a Prophet model tuned for COVID time-series data.

    changepoint_prior_scale=0.05 makes the trend less flexible, reducing
    overfitting to individual wave peaks — appropriate for a planning tool.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed.")

    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        interval_width=0.95,   # 95% confidence interval
    )
    return model


def fit_and_forecast(
    df_prophet: pd.DataFrame,
    forecast_horizon_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit Prophet on historical data and produce a forward forecast.

    Parameters
    ----------
    df_prophet            : pd.DataFrame  Columns ['ds', 'y'].
    forecast_horizon_days : int           Days ahead to forecast.

    Returns
    -------
    forecast_df  : pd.DataFrame  Full Prophet forecast (historical + future).
    future_only  : pd.DataFrame  Only the forward-looking rows.
    """
    model = build_model()
    # Add floor/cap for logistic growth guard
    df_prophet = df_prophet.copy()
    df_prophet["y"] = df_prophet["y"].clip(lower=0.01)  # avoid exact zeros

    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=forecast_horizon_days)
    forecast = model.predict(future)

    # Clip negative lower bounds — case counts can't be negative
    forecast["yhat"]       = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    future_only = forecast[forecast["ds"] > df_prophet["ds"].max()].copy()

    logger.info(
        "Forecast complete. Horizon: %d days. Peak predicted: %.0f",
        forecast_horizon_days,
        future_only["yhat"].max(),
    )
    return forecast, future_only


def evaluate_model(
    df_prophet: pd.DataFrame,
    initial_days: int = 365,
    period_days: int = 30,
    horizon_days: int = 30,
) -> Dict[str, float]:
    """
    Run time-series cross-validation and return key accuracy metrics.

    Business interpretation:
      - MAE: average daily case mis-estimate (in absolute count)
      - MAPE: % error — useful for scaling across different countries
      - Coverage: % of actual values that fall within the 95% CI

    Parameters
    ----------
    initial_days : int  Minimum training period.
    period_days  : int  Spacing between cutoff points.
    horizon_days : int  Forecast horizon to evaluate.

    Returns
    -------
    dict  {'mae': float, 'mape': float, 'rmse': float, 'coverage': float}
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed.")

    model = build_model()
    model.fit(df_prophet)

    df_cv = cross_validation(
        model,
        initial=f"{initial_days} days",
        period=f"{period_days} days",
        horizon=f"{horizon_days} days",
        parallel="threads",
    )
    metrics = performance_metrics(df_cv)

    summary = {
        "mae":      round(metrics["mae"].mean(),  2),
        "mape":     round(metrics["mape"].mean() * 100, 2),   # as %
        "rmse":     round(metrics["rmse"].mean(), 2),
        "coverage": round(metrics["coverage"].mean() * 100, 2),
    }

    logger.info("Model evaluation — %s", summary)
    return summary


# ── Surge Detection ───────────────────────────────────────────────────────────

def detect_surge(
    forecast_df: pd.DataFrame,
    baseline_window_days: int = 14,
    surge_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag dates in the forecast where predicted cases exceed
    (surge_multiplier × recent baseline) — a simple operational alert rule.

    Business value: Hospitals can use this flag to trigger surge protocols
    (e.g., activating overflow capacity, recalling off-duty staff).

    Parameters
    ----------
    forecast_df       : pd.DataFrame  Output from fit_and_forecast().
    baseline_window   : int           Days used to compute the rolling baseline.
    surge_multiplier  : float         Threshold ratio (default: 50% above baseline).

    Returns
    -------
    pd.DataFrame  with an additional boolean column 'is_surge'.
    """
    df = forecast_df.copy()
    df["baseline"] = (
        df["yhat"]
        .rolling(window=baseline_window_days, min_periods=1)
        .mean()
        .shift(baseline_window_days)
    )
    df["is_surge"] = df["yhat"] > (df["baseline"] * surge_multiplier)
    return df


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_forecast_pipeline(
    df_prophet: pd.DataFrame,
    forecast_horizon_days: int = 30,
    run_eval: bool = False,
) -> Dict:
    """
    End-to-end pipeline: fit → forecast → surge detection → optional eval.

    Returns
    -------
    dict with keys:
        'forecast'   : full forecast dataframe
        'future'     : forward-looking rows only
        'surge'      : forecast with surge flags
        'metrics'    : evaluation metrics (if run_eval=True)
    """
    forecast, future = fit_and_forecast(df_prophet, forecast_horizon_days)
    surge = detect_surge(forecast)

    result = {"forecast": forecast, "future": future, "surge": surge}

    if run_eval and len(df_prophet) > 400:   # need enough data for CV
        result["metrics"] = evaluate_model(df_prophet)
    else:
        result["metrics"] = None

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_and_filter
    from src.preprocessing import prepare_forecasting

    df = load_and_filter()
    df_p = prepare_forecasting(df, "United States")
    results = run_forecast_pipeline(df_p, forecast_horizon_days=30, run_eval=False)
    print(results["future"][["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10))
    print("Surge days:", results["surge"]["is_surge"].sum())