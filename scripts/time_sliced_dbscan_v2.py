"""
Time-sliced DBSCAN with CONSISTENT cluster colors across panels.
Colors are assigned by cluster centroid position, so the same
physical building keeps the same color across all time windows.
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

INPUT = "data/kth_aggregated.csv"
FIG = "figures/fig_kth_time_sliced_v2.png"

EPS_M = 30.0
MIN_PTS = 3
SLICES = [
    ("Night\n(0-6)",    0, 6),
    ("Morning\n(6-12)", 6, 12),
    ("Afternoon\n(12-17)", 12, 17),
    ("Evening\n(17-22)", 17, 22),
    ("Late\n(22-24)",   22, 24),
]

def centroid(labels, X, cid):
    return X[labels == cid].mean(axis=0)

def main():
    df = pd.read_csv(INPUT)

    # Global cluster reference: run DBSCAN once on all data to establish
    # canonical cluster IDs sorted by position (used for consistent coloring)
    X_all = df[["x","y"]].values
    labels_all = DBSCAN(eps=EPS_M, min_samples=MIN_PTS,
                        algorithm="kd_tree", leaf_size=30,
                        n_jobs=-1).fit_predict(X_all)
    ref_ids = sorted(set(labels_all) - {-1})
    ref_centroids = np.array([centroid(labels_all, X_all, c) for c in ref_ids])
    # order reference clusters by size (largest first)
    sizes = [(cid, (labels_all == cid).sum()) for cid in ref_ids]
    sizes.sort(key=lambda t: -t[1])
    ordered_ids = [cid for cid, _ in sizes]
    ordered_centroids = np.array([centroid(labels_all, X_all, c) for c in ordered_ids])

    def nearest_ref_color(xy):
        d = np.linalg.norm(ordered_centroids - xy, axis=1)
        return int(np.argmin(d))

    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 5, figsize=(24, 6), sharex=True, sharey=True)
    for i, (label, h0, h1) in enumerate(SLICES):
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)].copy()
        X = sub[["x","y"]].values
        labels = DBSCAN(eps=EPS_M, min_samples=MIN_PTS,
                        algorithm="kd_tree", leaf_size=30,
                        n_jobs=-1).fit_predict(X)
        sub["cluster"] = labels

        ax = axes[i]
        for cid in sorted(set(labels) - {-1}):
            sub_c = sub[sub["cluster"] == cid]
            cen = sub_c[["x","y"]].mean().values
            color_idx = nearest_ref_color(cen)
            ax.scatter(sub_c["x"], sub_c["y"], s=10,
                       color=cmap(color_idx % 20), alpha=0.75)
        ax.set_xlim(21000, 22500)
        ax.set_ylim(31500, 34000)
        ax.set_title(f"{label}\n{len(set(labels)-{-1})} clusters", fontsize=10)
        ax.set_xlabel("X (m)", fontsize=8)
        if i == 0:
            ax.set_ylabel("Y (m)", fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("KTH campus WiFi — time-sliced DBSCAN (consistent colors)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG, dpi=180)
    plt.close()
    print(f"Saved {FIG}")

if __name__ == "__main__":
    main()
