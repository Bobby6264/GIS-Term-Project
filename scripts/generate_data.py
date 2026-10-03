"""
Synthetic Campus Event Generator
--------------------------------
Generates 5000 campus events across 3 known hotspots + 500 noise events.
Each event has: event_id, lon, lat, day (0-29), hour (0-23), type.

The 3 hotspots have realistic temporal patterns:
  - Library:    peaks during study hours (10-11am, 2-4pm)
  - Cafeteria:  peaks at meal times (12-1pm, 7-8pm)
  - Sports:     peaks early morning + evening (6-7am, 5-6pm)

This ground truth lets us validate whether DBSCAN / ST-DBSCAN recover
the correct clusters.
"""

import numpy as np
import pandas as pd
import os

# ---- reproducibility ----
np.random.seed(42)

# ---- config ----
N_DAYS = 30
OUTPUT_PATH = os.path.join("data", "campus_events.csv")
HOTSPOTS = [
    {
        "name": "Library",
        "lon": 88.0000, "lat": 22.5000,
        "n": 1500,
        "spread": 0.00025,                 # tighter: ~25m std
        "peak_hours": [10, 11, 14, 15, 16],
    },
    {
        "name": "Cafeteria",
        "lon": 88.0045, "lat": 22.5030,    # further away
        "n": 1800,
        "spread": 0.00030,                 # ~30m
        "peak_hours": [12, 13, 19, 20],
    },
    {
        "name": "Sports Complex",
        "lon": 87.9955, "lat": 22.4965,    # further away
        "n": 1200,
        "spread": 0.00035,                 # ~35m
        "peak_hours": [6, 7, 17, 18],
    },
]
# ---- noise events: uniformly scattered, no pattern ----
N_NOISE = 500
NOISE_LON_RANGE = (87.993, 88.007)
NOISE_LAT_RANGE = (22.494, 22.506)


def generate_hotspot_events(hotspot, start_id):
    """Generate N events clustered around one hotspot with temporal peaks."""
    n = hotspot["n"]
    lon = np.random.normal(hotspot["lon"], hotspot["spread"], n)
    lat = np.random.normal(hotspot["lat"], hotspot["spread"], n)
    hours = np.random.choice(hotspot["peak_hours"], n)
    days = np.random.randint(0, N_DAYS, n)

    return pd.DataFrame({
        "event_id": range(start_id, start_id + n),
        "lon": lon,
        "lat": lat,
        "day": days,
        "hour": hours,
        "type": hotspot["name"],
    })


def generate_noise_events(start_id):
    """Generate uniformly scattered noise events."""
    n = N_NOISE
    return pd.DataFrame({
        "event_id": range(start_id, start_id + n),
        "lon": np.random.uniform(*NOISE_LON_RANGE, n),
        "lat": np.random.uniform(*NOISE_LAT_RANGE, n),
        "day": np.random.randint(0, N_DAYS, n),
        "hour": np.random.randint(0, 24, n),
        "type": "other",
    })


def main():
    os.makedirs("data", exist_ok=True)

    frames = []
    next_id = 0

    for h in HOTSPOTS:
        df_h = generate_hotspot_events(h, next_id)
        frames.append(df_h)
        next_id += h["n"]

    df_noise = generate_noise_events(next_id)
    frames.append(df_noise)

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Generated {len(df)} events → {OUTPUT_PATH}")
    print("\nDistribution by type:")
    print(df["type"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nHour distribution (sample):")
    print(df["hour"].value_counts().sort_index().head(10))


if __name__ == "__main__":
    main()