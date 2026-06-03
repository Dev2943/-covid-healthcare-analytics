"""
prepare_data.py
---------------
Runs ONCE at Docker BUILD time. Downloads the full OWID dataset,
trims it to only the columns and aggregate-free rows the app needs,
and saves a compact CSV to a FIXED absolute path (/app/data/).

This keeps the runtime image small and makes startup instant —
no download, no 400k-row parse, no out-of-memory kill.
"""
import os
import pandas as pd
import urllib.request

URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
OUT_DIR = "/app/data"
OUT_PATH = os.path.join(OUT_DIR, "owid_covid_data.csv")

COLUMNS = [
    "iso_code", "continent", "location", "date",
    "total_cases", "new_cases", "new_cases_smoothed",
    "total_deaths", "new_deaths", "new_deaths_smoothed",
    "icu_patients", "icu_patients_per_million",
    "hosp_patients", "hosp_patients_per_million",
    "population", "hospital_beds_per_thousand",
]

AGGREGATES = [
    "World", "Africa", "Asia", "Europe", "European Union",
    "North America", "South America", "Oceania",
    "High income", "Upper middle income", "Lower middle income", "Low income",
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading full OWID dataset (build-time only)…")
    tmp = "/tmp/owid_full.csv"
    urllib.request.urlretrieve(URL, tmp)

    print("Trimming columns + dropping aggregate rows…")
    df = pd.read_csv(tmp, low_memory=False)
    cols = [c for c in COLUMNS if c in df.columns]
    df = df[cols]
    df = df[~df["location"].isin(AGGREGATES)]

    # Downcast numerics to shrink memory footprint
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")

    df.to_csv(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Saved compact dataset: {len(df):,} rows, {size_mb:.1f} MB → {OUT_PATH}")
    os.remove(tmp)

if __name__ == "__main__":
    main()
