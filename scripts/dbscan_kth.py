import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

INPUT       = "data/kth_events.csv"
FIG_KDDIST  = "figures/fig_kth_kdistance.png"
FIG_CLUST   = "figures/fig_kth_dbscan_clusters.png"
METRICS     = "data/metrics_kth_dbscan.csv"

MIN_PTS     = 20
SAMPLE_FRAC = 1.0
EPS_M       = 50.0

def plot_kdistance(X, k, out_path, max_points=50_000):
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
    dists, _ = nbrs.kneighbors(X_sample)
    kdist = np.sort(dists[:, k-1])
    plt.figure(figsize=(8,5))
    plt.plot(kdist, linewidth=2)
    plt.xlabel(f"Points sorted by distance to {k}-th neighbor")
    plt.ylabel(f"Distance to {k}-th neighbor (m)")
    plt.title("KTH k-Distance Graph")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")
    for pct in [50, 75, 90, 95, 97]:
        print(f"  {pct}th percentile: {np.percentile(kdist, pct):.1f} m")

def main():
    os.makedirs("figures", exist_ok=True)
    print(f"Loading {INPUT} ...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} events")
    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
        print(f"  Sampled to {len(df):,}")
    X = df[["x","y"]].values
    plot_kdistance(X, MIN_PTS, FIG_KDDIST)
    print(f"\nRunning DBSCAN eps={EPS_M} m, minPts={MIN_PTS} ...")
    t0 = time.time()
    labels = DBSCAN(eps=EPS_M, min_samples=MIN_PTS,
                    algorithm="ball_tree", leaf_size=40, n_jobs=-1).fit_predict(X)
    rt = time.time() - t0
    n_c = len(set(labels)) - (1 if -1 in labels else 0)
    n_n = int((labels==-1).sum())
    print(f"  clusters: {n_c}")
    print(f"  noise:    {n_n:,} ({100*n_n/len(labels):.1f}%)")
    print(f"  runtime:  {rt:.1f} s")
    sil = None
    mask = labels != -1
    if len(set(labels[mask])) > 1 and mask.sum() > 100:
        sil = round(silhouette_score(X[mask], labels[mask]), 3)
        print(f"  silhouette: {sil}")
    df["cluster"] = labels
    pd.DataFrame([{
        "dataset":"KTH","method":"DBSCAN","eps_m":EPS_M,"min_pts":MIN_PTS,
        "n_events":len(df),"n_clusters":n_c,"n_noise":n_n,
        "noise_pct":round(100*n_n/len(df),2),"silhouette":sil,
        "runtime_s":round(rt,1)
    }]).to_csv(METRICS, index=False)
    print(f"Saved {METRICS}")
    df.to_csv("data/kth_events_dbscan.csv", index=False)
    print("Saved data/kth_events_dbscan.csv")
    plot_df = df.sample(min(200_000, len(df)), random_state=42)
    plt.figure(figsize=(11,9))
    noise = plot_df[plot_df["cluster"]==-1]
    plt.scatter(noise["x"], noise["y"], s=1, c="lightgray", alpha=0.4, label="Noise")
    cmap = plt.get_cmap("tab20")
    top = df["cluster"].value_counts().head(20).index.tolist()
    top = [c for c in top if c != -1]
    for i, cid in enumerate(top):
        sub = plot_df[plot_df["cluster"]==cid]
        plt.scatter(sub["x"], sub["y"], s=1.5, color=cmap(i%20),
                    label=f"C{cid} ({len(df[df['cluster']==cid]):,})", alpha=0.7)
    plt.title(f"KTH campus WiFi — DBSCAN ({n_c} clusters)")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.legend(markerscale=5, fontsize=7, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_CLUST, dpi=180)
    plt.close()
    print(f"Saved {FIG_CLUST}")

if __name__ == "__main__":
    main()