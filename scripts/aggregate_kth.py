"""
Aggregate KTH WiFi events into (AP, hour, day) counts.
Much smaller, still spatio-temporal, faster to cluster.
"""

import pandas as pd
import os

INP = "data/kth_events.csv"
OUT = "data/kth_aggregated.csv"

print(f"Loading {INP} ...")
df = pd.read_csv(INP)
print(f"  {len(df):,} raw events")

# aggregate: count events per (AP, hour, day)
agg = (
    df.groupby(["AP", "x", "y", "hour", "day"])
      .size()
      .reset_index(name="count")
)
print(f"  {len(agg):,} aggregated rows")
print(f"  count distribution:\n{agg['count'].describe()}")

agg.to_csv(OUT, index=False)
print(f"Saved {OUT}")
