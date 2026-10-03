"""
ST-DBSCAN: Spatio-Temporal DBSCAN
---------------------------------
Clusters events that are close in BOTH space AND time.

Approach:
  1. Project lat/lon to meters (same as before)
  2. Convert hour to cyclical representation? -> NO, we want linear time here
     (a 23:00 event and a 01:00 event should NOT be close unless we do cyclic)
     For this project, treat hour as linear 0..23.
  3. Combine [x, y, hour_scaled] into one feature space.
  4. Scale so that eps_time (hours) and eps_spatial (meters) have comparable
     influence.
  5. Run DBSCAN on the combined space.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

INPUT_PATH = "data/campus_events.csv"
OUT_CSV    = "data/campus_events_stdbscan.csv"
OUT_FIG    = "figures/fig_stdbscan_clusters.png"
OUT_METRIC = "data/metrics_stdbscan.csv"


# ---------- projection ----------
def project_to_meters(df):
    lat0, lon0 = df["lat"].mean(), df["lon"].mean()
    df["x"] = (df["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0))
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


# ---------- ST-DBSCAN core ----------
def st_dbscan(df, eps_spatial_m=75.0, eps_time_hr=3.0, min_pts=10):
    """
    Run DBSCAN in a 3D feature space [x, y, hour].

    To make sure eps has a consistent meaning, we scale the time axis so
    that eps_time_hr in time corresponds to roughly eps_spatial_m in space.

    Concretely:
      - Space features are in meters.
      - Time feature is scaled:  hour * (eps_spatial_m / eps_time_hr)
      - Then a single eps = eps_spatial_m is used for all 3 axes.
    """
    scale = eps_spatial_m / eps_time_hr      # meters per hour equivalent

    feats = np.column_stack([
        df["x"].values,
        df["y"].values,
        df["hour"].values * scale,
    ])

    labels = DBSCAN(eps=eps_spatial_m, min_samples=min_pts).fit_predict(feats)
    return labels


# ---------- metrics ----------
def compute_metrics(df, labels, eps_spatial, eps_time, min_pts, runtime):
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    sil = None
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        # Silhouette on the scaled feature space
        scale = eps_spatial / eps_time
        X = np.column_stack([
            df.loc[mask, "x"],
            df.loc[mask, "y"],
            df.loc[mask, "hour"] * scale,
        ])
        sil = silhouette_score(X, labels[mask])

    return {
        "eps_spatial_m": eps_spatial,
        "eps_time_hr": eps_time,
        "min_pts": min_pts,
        "n_total": n_total,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / n_total, 2),
        "silhouette": round(sil, 3) if sil is not None else None,
        "runtime_s": round(runtime, 2),
    }


# ---------- plotting ----------
def plot_stdbscan(df, out_path):
    """
    Two panels side by side:
      Left  : spatial view (x, y) — colored by ST cluster
      Right : temporal view (hour vs cluster id) — shows WHEN each cluster occurs
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---- left: spatial ----
    ax = axes[0]
    noise = df[df["cluster"] == -1]
    ax.scatter(noise["x"], noise["y"], s=8, c="lightgray",
               label=f"Noise ({len(noise)})", alpha=0.6)

    clusters = df[df["cluster"] != -1]
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(sorted(clusters["cluster"].unique())):
        sub = clusters[clusters["cluster"] == cid]
        ax.scatter(sub["x"], sub["y"], s=10, color=cmap(i % 20),
                   label=f"C{cid} ({len(sub)})", alpha=0.75)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Spatial view — ST-DBSCAN clusters")
    ax.legend(markerscale=2, fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # ---- right: temporal ----
    ax = axes[1]
    for i, cid in enumerate(sorted(clusters["cluster"].unique())):
        sub = clusters[clusters["cluster"] == cid]
        ax.hist(sub["hour"], bins=range(25), alpha=0.5,
                color=cmap(i % 20), label=f"C{cid}")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Event count")
    ax.set_title("Temporal view — when each cluster occurs")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ---------- main ----------
def main():
    os.makedirs("figures", exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = project_to_meters(df)
    print(f"Loaded {len(df)} events")

    EPS_SPATIAL = 50.0    # meters (same as spatial DBSCAN)
    EPS_TIME    = 3.0     # hours
    MIN_PTS     = 10

    t0 = time.time()
    labels = st_dbscan(df, EPS_SPATIAL, EPS_TIME, MIN_PTS)
    runtime = time.time() - t0

    df["cluster"] = labels

    metrics = compute_metrics(df, labels, EPS_SPATIAL, EPS_TIME, MIN_PTS, runtime)
    print("\n📊 ST-DBSCAN metrics:")
    for k, v in metrics.items():
        print(f"  {k:>15}: {v}")

    print("\nCluster size distribution:")
    print(df["cluster"].value_counts().sort_index().to_string())

    pd.DataFrame([metrics]).to_csv(OUT_METRIC, index=False)
    df.to_csv(OUT_CSV, index=False)

    plot_stdbscan(df, OUT_FIG)


if __name__ == "__main__":
    main()