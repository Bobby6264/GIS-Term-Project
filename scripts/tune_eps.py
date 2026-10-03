"""
Parameter sweep for DBSCAN eps.
Shows how cluster count + noise % change as eps varies.
Useful for justifying eps in the presentation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

df = pd.read_csv("data/campus_events.csv")

# same projection as dbscan_spatial.py
lat0, lon0 = df["lat"].mean(), df["lon"].mean()
df["x"] = (df["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0))
df["y"] = (df["lat"] - lat0) * 110_540
X = df[["x", "y"]].values

rows = []
for eps in [30, 40, 50, 60, 70, 75, 80, 90, 100, 120]:
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    rows.append({
        "eps_m": eps,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / len(labels), 2),
    })

df_out = pd.DataFrame(rows)
print(df_out.to_string(index=False))
df_out.to_csv("data/eps_sweep.csv", index=False)
print("\nSaved data/eps_sweep.csv")