"""
ST-DBSCAN on aggregated KTH data.
Uses ONLY (x, y, hour-of-day) — collapses days so daily patterns emerge.
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

INPUT    = "data/kth_aggregated.csv"
OUT_CSV  = "data/kth_aggregated_stdbscan.csv"
OUT_FIG  = "figures/fig_kth_stdbscan_clusters.png"
METRICS  = "data/metrics_kth_stdbscan.csv"

EPS_SPATIAL = 30.0
EPS_TIME    = 2.0      # tighter: 2-hour window
MIN_PTS     = 3       # stricter: needs more neighbors to form a cluster

def main():
    os.makedirs("figures", exist_ok=True)
    print(f"Loading {INPUT} ...", flush=True)
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} rows", flush=True)

    # KEY CHANGE: collapse day -> use hour-of-day only
    df2 = df.groupby(["AP","x","y","hour"], as_index=False)["count"].sum()
    print(f"  after day-collapse: {len(df2):,} rows", flush=True)

    scale = EPS_SPATIAL / EPS_TIME
    feats = np.column_stack([
        df2["x"].values,
        df2["y"].values,
        df2["hour"].values * scale,
    ])

    print(f"\nRunning ST-DBSCAN (eps_spatial={EPS_SPATIAL} m, "
          f"eps_time={EPS_TIME} hr, minPts={MIN_PTS}) ...", flush=True)
    t0 = time.time()
    labels = DBSCAN(eps=EPS_SPATIAL, min_samples=MIN_PTS,
                    algorithm="kd_tree", leaf_size=30, n_jobs=-1).fit_predict(feats)
    rt = time.time() - t0

    n_c = len(set(labels)) - (1 if -1 in labels else 0)
    n_n = int((labels == -1).sum())
    print(f"  clusters: {n_c}", flush=True)
    print(f"  noise:    {n_n:,} ({100*n_n/len(labels):.1f}%)", flush=True)
    print(f"  runtime:  {rt:.1f} s", flush=True)

    sil = None
    mask = labels != -1
    if len(set(labels[mask])) > 1 and mask.sum() > 1000:
        idx = np.random.choice(np.where(mask)[0], 10_000, replace=False)
        sil = round(silhouette_score(feats[idx], labels[idx]), 3)
        print(f"  silhouette (sample): {sil}", flush=True)

    df2["cluster"] = labels
    pd.DataFrame([{
        "dataset":"KTH_aggregated","method":"ST-DBSCAN",
        "eps_spatial_m":EPS_SPATIAL,"eps_time_hr":EPS_TIME,"min_pts":MIN_PTS,
        "n_events":len(df2),"n_clusters":n_c,"n_noise":n_n,
        "noise_pct":round(100*n_n/len(df2),2),
        "silhouette":sil,"runtime_s":round(rt,1)
    }]).to_csv(METRICS, index=False)
    print(f"Saved {METRICS}", flush=True)
    df2.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}", flush=True)

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    noise = df2[df2["cluster"] == -1]
    ax.scatter(noise["x"], noise["y"], s=8, c="lightgray",
               alpha=0.4, label=f"Noise ({len(noise):,})")
    cmap = plt.get_cmap("tab20")
    top = df2["cluster"].value_counts().head(20).index.tolist()
    top = [c for c in top if c != -1]
    for i, cid in enumerate(top):
        sub = df2[df2["cluster"] == cid]
        ax.scatter(sub["x"], sub["y"], s=15, color=cmap(i % 20),
                   label=f"C{cid} ({len(sub):,})", alpha=0.75)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"ST-DBSCAN spatial view ({n_c} clusters)")
    ax.legend(markerscale=2, fontsize=6, loc="best", ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, cid in enumerate(top[:12]):
        sub = df2[df2["cluster"] == cid]
        ax.hist(sub["hour"], bins=range(25), alpha=0.5,
                color=cmap(i % 20), label=f"C{cid}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("AP-hour count")
    ax.set_title("Temporal profile per cluster")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=180)
    plt.close()
    print(f"Saved {OUT_FIG}", flush=True)

    print("\nCluster size distribution:", flush=True)
    print(df2["cluster"].value_counts().head(20).to_string(), flush=True)
    print("\n✅ DONE", flush=True)

if __name__ == "__main__":
    main()
