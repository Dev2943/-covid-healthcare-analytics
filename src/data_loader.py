"""
data_loader.py
--------------
Downloads and caches the Our World in Data COVID-19 dataset.
Source: https://github.com/owid/covid-19-data
Dataset is updated daily and contains 67+ variables per country.
"""

import os
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "owid_covid_data.csv")

# Columns we care about for this project
COLUMNS_OF_INTEREST = [
    "iso_code", "continent", "location", "date",
    "total_cases", "new_cases", "new_cases_smoothed",
    "total_deaths", "new_deaths", "new_deaths_smoothed",
    "icu_patients", "icu_patients_per_million",
    "hosp_patients", "hosp_patients_per_million",
    "new_vaccinations", "total_vaccinations_per_hundred",
    "population", "population_density",
    "median_age", "gdp_per_capita",
    "hospital_beds_per_thousand",
]


def download_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download OWID COVID data if not cached, else load from cache.

    Parameters
    ----------
    force_refresh : bool
        If True, re-download even if a cached copy exists.

    Returns
    -------
    pd.DataFrame
        Raw OWID COVID-19 dataset.
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force_refresh:
        logger.info("Loading cached data from %s", CACHE_PATH)
        return pd.read_csv(CACHE_PATH, low_memory=False)

    logger.info("Downloading COVID-19 data from Our World in Data ...")
    try:
        response = requests.get(OWID_URL, timeout=60)
        response.raise_for_status()
        with open(CACHE_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Download complete. Saved to %s", CACHE_PATH)
        return pd.read_csv(CACHE_PATH, low_memory=False)
    except requests.RequestException as e:
        logger.error("Failed to download data: %s", e)
        raise


def load_and_filter(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the dataset and return only the columns relevant to our analysis.

    Returns
    -------
    pd.DataFrame
        Filtered and type-cast dataframe.
    """
    df = download_data(force_refresh=force_refresh)

    # Keep only columns that exist in this version of the dataset
    available_cols = [c for c in COLUMNS_OF_INTEREST if c in df.columns]
    df = df[available_cols].copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Drop OWID aggregate rows (continents, income groups, World)
    aggregate_locations = [
        "World", "Africa", "Asia", "Europe", "European Union",
        "North America", "South America", "Oceania",
        "High income", "Upper middle income", "Lower middle income", "Low income",
    ]
    df = df[~df["location"].isin(aggregate_locations)]

    logger.info(
        "Loaded %d rows across %d countries from %s to %s",
        len(df),
        df["location"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


if __name__ == "__main__":
    df = load_and_filter()
    print(df.head())
    print(df.dtypes)
