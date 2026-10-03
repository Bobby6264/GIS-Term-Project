"""
Spatial DBSCAN on Campus Events
-------------------------------
1. Load synthetic events
2. Project lat/lon -> local meters (equirectangular approximation)
3. k-distance graph to pick eps
4. Run DBSCAN
5. Compute metrics (cluster count, noise %, silhouette)
6. Save cluster plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# ---------------- config ----------------
INPUT_PATH   = "data/campus_events.csv"
FIG_KDDIST   = "figures/fig_kdistance.png"
FIG_CLUSTERS = "figures/fig_dbscan_clusters.png"
METRICS_PATH = "data/metrics_dbscan.csv"

MIN_PTS = 10          # min samples to form a dense region
K_FOR_KDDIST = MIN_PTS


# ---------------- 1. load ----------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} events from {path}")
    return df


# ---------------- 2. project to meters ----------------
def project_to_meters(df):
    """
    Equirectangular projection centered on the data's mean lat/lon.
    1 degree lat  ~ 110,540 m
    1 degree lon  ~ 111,320 * cos(lat) m
    Good enough for small (<10 km) areas.
    """
    lat0 = df["lat"].mean()
    lon0 = df["lon"].mean()

    df["x"] = (df["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0))
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


# ---------------- 3. k-distance graph ----------------
def plot_kdistance(X, k, out_path):
    """
    Plot sorted distance to k-th nearest neighbor.
    The 'knee' of this curve is a good eps.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    kdist = np.sort(dists[:, k - 1])

    plt.figure(figsize=(8, 5))
    plt.plot(kdist, linewidth=2)
    plt.xlabel(f"Points sorted by distance to {k}-th neighbor")
    plt.ylabel(f"Distance to {k}-th neighbor (meters)")
    plt.title("k-Distance Graph — knee ≈ good eps")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    # Print a few candidate eps values around the knee (90-99th percentile)
    for pct in [90, 95, 97, 99]:
        print(f"  {pct}th percentile distance: {np.percentile(kdist, pct):.1f} m")


# ---------------- 4. run DBSCAN ----------------
def run_dbscan(X, eps, min_pts):
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
    return db.labels_


# ---------------- 5. metrics ----------------
def compute_metrics(df, labels, eps, min_pts, runtime_s):
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    sil = None
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        sil = silhouette_score(df.loc[mask, ["x", "y"]], labels[mask])

    return {
        "eps_m": eps,
        "min_pts": min_pts,
        "n_total": n_total,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / n_total, 2),
        "silhouette": round(sil, 3) if sil is not None else None,
        "runtime_s": round(runtime_s, 2),
    }


# ---------------- 6. plot clusters ----------------
def plot_clusters(df, out_path):
    plt.figure(figsize=(9, 7))

    noise = df[df["cluster"] == -1]
    plt.scatter(noise["x"], noise["y"], s=8, c="lightgray",
                label=f"Noise ({len(noise)})", alpha=0.6)

    clusters = df[df["cluster"] != -1]
    n_clusters = clusters["cluster"].nunique()
    cmap = plt.get_cmap("tab10")
    for i, cid in enumerate(sorted(clusters["cluster"].unique())):
        sub = clusters[clusters["cluster"] == cid]
        plt.scatter(sub["x"], sub["y"], s=10,
                    color=cmap(i % 10),
                    label=f"Cluster {cid} ({len(sub)})",
                    alpha=0.75)

    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(f"Spatial DBSCAN — {n_clusters} clusters found")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ---------------- main ----------------
def main():
    import time
    os.makedirs("figures", exist_ok=True)

    df = load_data(INPUT_PATH)
    df = project_to_meters(df)

    X = df[["x", "y"]].values

    # k-distance graph
    plot_kdistance(X, K_FOR_KDDIST, FIG_KDDIST)

    # ---- choose eps based on the knee you saw above ----
    # Start with something ~1.5x the 95th percentile distance.
    eps = 40.0          # meters (tune after looking at k-distance output)

    t0 = time.time()
    labels = run_dbscan(X, eps=eps, min_pts=MIN_PTS)
    runtime = time.time() - t0

    df["cluster"] = labels

    metrics = compute_metrics(df, labels, eps, MIN_PTS, runtime)
    print("\n📊 DBSCAN metrics:")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")

    # save metrics
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    print(f"\nSaved {METRICS_PATH}")

    # save clustered CSV for later steps
    df.to_csv("data/campus_events_dbscan.csv", index=False)
    print("Saved data/campus_events_dbscan.csv")

    plot_clusters(df, FIG_CLUSTERS)


if __name__ == "__main__":
    main()