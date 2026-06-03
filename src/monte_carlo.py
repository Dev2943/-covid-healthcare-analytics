"""
monte_carlo.py
--------------
Monte Carlo scenario simulation for COVID case forecasting.

Why this belongs in a healthcare planning tool
-----------------------------------------------
A single Prophet point forecast tells administrators the *expected* case
load. But capacity planning is about tail risk: "how bad could it
plausibly get?" This module simulates many possible trajectories and
reports risk bands — the epidemiological analogue of Value-at-Risk.

  - Expected path        → staff/bed planning baseline
  - 95th percentile path → surge capacity that must be ON STANDBY
  - "Case-at-Risk" (CaR) → the 95% worst-case peak, one number for execs

This reuses the same geometric-random-walk + variance logic used in
quantitative finance Monte Carlo option pricing, applied to case growth.
"""

import numpy as np
import pandas as pd
from typing import Dict


def simulate_trajectories(
    df_prophet: pd.DataFrame,
    horizon_days: int = 30,
    n_simulations: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate `n_simulations` possible future case trajectories using a
    geometric random walk calibrated to recent historical volatility.

    Parameters
    ----------
    df_prophet    : DataFrame with columns ['ds', 'y'] (Prophet format).
    horizon_days  : days to simulate forward.
    n_simulations : number of Monte Carlo paths.
    seed          : RNG seed for reproducibility.

    Returns
    -------
    np.ndarray  shape (n_simulations, horizon_days) of simulated case counts.
    """
    np.random.seed(seed)

    y = df_prophet["y"].values
    y = y[y > 0]
    if len(y) < 30:
        raise ValueError("Need at least 30 days of positive data to calibrate.")

    # Daily log-returns of the case series → drift (mu) and volatility (sigma)
    log_returns = np.diff(np.log(y[-90:]))          # last 90 days
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)

    start_value = y[-1]
    trajectories = np.zeros((n_simulations, horizon_days))

    for i in range(n_simulations):
        price = start_value
        for t in range(horizon_days):
            # Geometric random walk with drift — same engine as GBM option pricing
            shock = np.random.normal(mu, sigma)
            # Mild mean reversion so paths don't explode
            shock = np.clip(shock, -0.5, 0.5)
            price = max(price * np.exp(shock), 0)
            trajectories[i, t] = price

    return trajectories


def summarize_simulation(
    trajectories: np.ndarray,
    last_date: pd.Timestamp,
) -> Dict:
    """
    Reduce raw trajectories to decision-ready risk bands.

    Returns
    -------
    dict with:
      'dates'        : forecast dates
      'expected'     : median path (50th percentile)
      'lower_50'     : 25th percentile
      'upper_50'     : 75th percentile
      'lower_95'     : 2.5th percentile
      'upper_95'     : 97.5th percentile  (the surge-planning line)
      'case_at_risk' : 95th-percentile peak across the horizon (one KPI)
    """
    horizon = trajectories.shape[1]
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    summary = {
        "dates":        dates,
        "expected":     np.percentile(trajectories, 50, axis=0),
        "lower_50":     np.percentile(trajectories, 25, axis=0),
        "upper_50":     np.percentile(trajectories, 75, axis=0),
        "lower_95":     np.percentile(trajectories, 2.5, axis=0),
        "upper_95":     np.percentile(trajectories, 97.5, axis=0),
        "case_at_risk": float(np.percentile(trajectories.max(axis=1), 95)),
    }
    return summary


def run_monte_carlo(
    df_prophet: pd.DataFrame,
    horizon_days: int = 30,
    n_simulations: int = 1000,
) -> Dict:
    """End-to-end: simulate trajectories and return decision-ready summary."""
    trajectories = simulate_trajectories(df_prophet, horizon_days, n_simulations)
    last_date = pd.to_datetime(df_prophet["ds"].max())
    summary = summarize_simulation(trajectories, last_date)
    summary["trajectories"] = trajectories      # keep raw for fan-chart plotting
    summary["n_simulations"] = n_simulations
    return summary


if __name__ == "__main__":
    # Quick self-test with synthetic data
    dates = pd.date_range("2022-01-01", periods=200)
    cases = 1000 * np.exp(np.cumsum(np.random.normal(0.005, 0.05, 200)))
    df = pd.DataFrame({"ds": dates, "y": cases})

    result = run_monte_carlo(df, horizon_days=30, n_simulations=1000)
    print(f"Expected peak (30d): {result['expected'].max():.0f}")
    print(f"95% worst-case peak (Case-at-Risk): {result['case_at_risk']:.0f}")
    print(f"Simulated {result['n_simulations']} trajectories")
