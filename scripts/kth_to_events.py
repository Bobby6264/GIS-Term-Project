"""
Convert raw KTH/Campus data → standardized events.csv format.

Input:
  data/kth/2014_01.csv.gz          (5.6M rows: timestamp, client, AP)
  data/kth/APlocations.txt         (AP name -> x, y, floor)

Output:
  data/kth_events.csv              (timestamp, x, y, hour, day, ap, floor)
"""

import os
import pandas as pd

CSV_IN     = "data/kth/2014_01.csv.gz"
AP_IN      = "data/kth/APlocations.txt"
OUT        = "data/kth_events.csv"

# Optional: limit to keep runtime manageable
MAX_ROWS   = None      # None = use all 5.6M. Set to 1_000_000 for a faster test.


def load_ap_locations(path):
    """Load AP -> (x, y, floor) mapping."""
    ap = pd.read_csv(path)
    ap.columns = [c.strip() for c in ap.columns]   # strip whitespace in headers
    print(f"Loaded {len(ap)} AP locations")
    print(f"  Columns: {ap.columns.tolist()}")
    print(ap.head(3).to_string())
    return ap


def load_events(path, nrows=None):
    """Load wifi association events."""
    df = pd.read_csv(path, nrows=nrows)
    print(f"\nLoaded {len(df):,} events")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"]  = df["timestamp"].dt.day
    return df


def merge(df_events, df_ap):
    """Join events with AP coordinates."""
    df = df_events.merge(
        df_ap,
        left_on="AP",
        right_on="AP",
        how="inner",
    )
    print(f"\nAfter merge: {len(df):,} events (dropped {len(df_events) - len(df):,} unmatched APs)")

    # rename for pipeline compatibility
    df = df.rename(columns={
        "x_coordinate(m)": "x",
        "y_coordinate(m)": "y",
    })

    # sanity check
    print("\nSample merged rows:")
    print(df[["timestamp", "AP", "x", "y", "hour", "day"]].head().to_string())

    return df


def main():
    os.makedirs("data", exist_ok=True)

    ap = load_ap_locations(AP_IN)
    events = load_events(CSV_IN, nrows=MAX_ROWS)

    df = merge(events, ap)

    # keep only useful columns
    keep = ["timestamp", "x", "y", "hour", "day", "AP", "floor"]
    df = df[keep]

    # deduplicate: same client connecting to same AP within a second is one event
    # (we dropped `client` already, so just drop exact dups)
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "AP"]).reset_index(drop=True)
    print(f"\nDropped {before - len(df):,} duplicate (timestamp, AP) rows")

    df.to_csv(OUT, index=False)
    print(f"\n✅ Saved {len(df):,} events → {OUT}")


if __name__ == "__main__":
    main()