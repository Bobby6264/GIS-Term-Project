"""
Download Chicago crime data (2023 sample).
Uses the Socrata API — no login needed.
"""

import os
import requests
import pandas as pd

OUT_PATH = "data/chicago_crime_sample.csv"
N_ROWS = 50000

URL = (
    "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
    "?$where=year=2023"
    f"&$limit={N_ROWS}"
    "&$select=id,primary_type,date,latitude,longitude"
)

def main():
    os.makedirs("data", exist_ok=True)

    print("Downloading Chicago crime data (2023)...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()

    with open(OUT_PATH, "wb") as f:
        f.write(r.content)

    df = pd.read_csv(OUT_PATH)
    print(f"Downloaded {len(df)} rows → {OUT_PATH}")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nNull lat/lon:", df[["latitude","longitude"]].isna().sum().to_dict())


if __name__ == "__main__":
    main()