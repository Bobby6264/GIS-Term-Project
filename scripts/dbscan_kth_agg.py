"""Spatial DBSCAN on aggregated KTH data — fast version."""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

INPUT   = "data/kth_aggregated.csv"
FIG_KD  = "figures/fig_kth_kdistance.png"
FIG_CL  = "figures/fig_kth_dbscan_clusters.png"
METRICS = "data/metrics_kth_dbscan.csv"

MIN_PTS = 3
EPS_M   = 30.0

def main():
    os.makedirs("figures", exist_ok=True)
    print(f"Loading {INPUT} ...", flush=True)
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} rows", flush=True)
    X = df[["x","y"]].values

    # quick k-distance plot (sample)
    print("Computing k-distance (sample of 30k)...", flush=True)
    idx = np.random.choice(len(X), min(30_000, len(X)), replace=False)
    nbrs = NearestNeighbors(n_neighbors=MIN_PTS).fit(X[idx])
    d, _ = nbrs.kneighbors(X[idx])
    kdist = np.sort(d[:, MIN_PTS-1])
    plt.figure(figsize=(8,5))
    plt.plot(kdist, linewidth=2)
    plt.xlabel(f"Points sorted by distance to {MIN_PTS}-th nn")
    plt.ylabel(f"Distance to {MIN_PTS}-th nn (m)")
    plt.title("KTH aggregated — k-Distance Graph")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_KD, dpi=180)
    plt.close()
    print(f"  saved {FIG_KD}", flush=True)
    for pct in [50, 75, 90, 95, 97, 99]:
        print(f"  {pct}th percentile: {np.percentile(kdist, pct):.2f} m", flush=True)

    # DBSCAN with kd_tree (much faster on 2D)
    print(f"\nRunning DBSCAN eps={EPS_M} m, minPts={MIN_PTS} ...", flush=True)
    t0 = time.time()
    labels = DBSCAN(eps=EPS_M, min_samples=MIN_PTS,
                    algorithm="kd_tree", leaf_size=30, n_jobs=-1).fit_predict(X)
    rt = time.time() - t0
    n_c = len(set(labels)) - (1 if -1 in labels else 0)
    n_n = int((labels == -1).sum())
    print(f"  clusters: {n_c}", flush=True)
    print(f"  noise:    {n_n:,} ({100*n_n/len(labels):.1f}%)", flush=True)
    print(f"  runtime:  {rt:.1f} s", flush=True)

    # silhouette on SAMPLE only
    sil = None
    mask = labels != -1
    if len(set(labels[mask])) > 1 and mask.sum() > 1000:
        idx2 = np.random.choice(np.where(mask)[0], 10_000, replace=False)
        print("Computing silhouette on 10k sample...", flush=True)
        sil = round(silhouette_score(X[idx2], labels[idx2]), 3)
        print(f"  silhouette (sample): {sil}", flush=True)

    df["cluster"] = labels
    pd.DataFrame([{
        "dataset":"KTH_aggregated","method":"DBSCAN",
        "eps_m":EPS_M,"min_pts":MIN_PTS,
        "n_events":len(df),"n_clusters":n_c,"n_noise":n_n,
        "noise_pct":round(100*n_n/len(df),2),
        "silhouette":sil,"runtime_s":round(rt,1)
    }]).to_csv(METRICS, index=False)
    print(f"Saved {METRICS}", flush=True)

    df.to_csv("data/kth_aggregated_dbscan.csv", index=False)
    print("Saved data/kth_aggregated_dbscan.csv", flush=True)

    # plot
    plt.figure(figsize=(11,9))
    noise = df[df["cluster"]==-1]
    plt.scatter(noise["x"], noise["y"], s=8, c="lightgray", alpha=0.4,
                label=f"Noise ({len(noise):,})")
    cmap = plt.get_cmap("tab20")
    top = df["cluster"].value_counts().head(20).index.tolist()
    top = [c for c in top if c != -1]
    for i, cid in enumerate(top):
        sub = df[df["cluster"]==cid]
        plt.scatter(sub["x"], sub["y"], s=15, color=cmap(i%20),
                    label=f"C{cid} ({len(sub):,})", alpha=0.75)
    plt.title(f"KTH campus WiFi — DBSCAN ({n_c} clusters)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_CL, dpi=180)
    plt.close()
    print(f"Saved {FIG_CL}", flush=True)
    print("\n✅ DONE", flush=True)

if __name__ == "__main__":
    main()
