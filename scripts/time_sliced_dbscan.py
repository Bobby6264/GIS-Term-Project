"""
Time-sliced spatial DBSCAN for KTH.
For each time-of-day window, run spatial DBSCAN separately.
This reveals DIFFERENT spatial hotspots at different times.
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

INPUT = "data/kth_aggregated.csv"
FIG = "figures/fig_kth_time_sliced.png"
METRICS = "data/metrics_kth_time_sliced.csv"

EPS_M = 30.0
MIN_PTS = 3
SLICES = [
    ("Night\n(0-6)",    0, 6),
    ("Morning\n(6-12)", 6, 12),
    ("Afternoon\n(12-17)", 12, 17),
    ("Evening\n(17-22)", 17, 22),
    ("Late\n(22-24)",   22, 24),
]

def main():
    os.makedirs("figures", exist_ok=True)
    print(f"Loading {INPUT} ...", flush=True)
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} rows", flush=True)

    fig, axes = plt.subplots(1, 5, figsize=(26, 7))
    rows = []

    for i, (label, h0, h1) in enumerate(SLICES):
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)].copy()
        if len(sub) < MIN_PTS:
            print(f"  {label}: too few rows", flush=True)
            continue
        X = sub[["x", "y"]].values

        print(f"\n{label}: {len(sub):,} rows", flush=True)
        t0 = time.time()
        labels = DBSCAN(eps=EPS_M, min_samples=MIN_PTS,
                        algorithm="kd_tree", leaf_size=30,
                        n_jobs=-1).fit_predict(X)
        rt = time.time() - t0
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        n_n = int((labels == -1).sum())
        print(f"  clusters: {n_c}, noise: {n_n} ({100*n_n/len(labels):.1f}%), "
              f"runtime: {rt:.1f}s", flush=True)

        sub["cluster"] = labels

        sil = None
        mask = labels != -1
        if len(set(labels[mask])) > 1 and mask.sum() > 500:
            idx = np.random.choice(np.where(mask)[0],
                                   min(5000, mask.sum()), replace=False)
            sil = round(silhouette_score(X[idx], labels[idx]), 3)
            print(f"  silhouette: {sil}", flush=True)

        rows.append({
            "slice": label.replace("\n", " "),
            "hours": f"{h0}-{h1}",
            "n_ap_hours": len(sub),
            "n_clusters": n_c,
            "n_noise": n_n,
            "noise_pct": round(100*n_n/len(sub), 2),
            "silhouette": sil,
            "runtime_s": round(rt, 2),
        })

        # plot on subplot
        ax = axes[i]
        noise = sub[sub["cluster"] == -1]
        ax.scatter(noise["x"], noise["y"], s=4, c="lightgray",
                   alpha=0.3, label=f"Noise")
        cmap = plt.get_cmap("tab10")
        top = sub["cluster"].value_counts().head(10).index.tolist()
        top = [c for c in top if c != -1]
        for j, cid in enumerate(top):
            pts = sub[sub["cluster"] == cid]
            ax.scatter(pts["x"], pts["y"], s=8, color=cmap(j % 10),
                       label=f"C{cid}({len(pts)})", alpha=0.75)
        ax.set_title(f"{label}\n{n_c} clusters", fontsize=10)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    plt.suptitle(f"KTH campus WiFi — time-sliced spatial DBSCAN (eps={EPS_M}m, minPts={MIN_PTS})",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG, dpi=180)
    plt.close()
    print(f"\nSaved {FIG}", flush=True)

    pd.DataFrame(rows).to_csv(METRICS, index=False)
    print(f"Saved {METRICS}", flush=True)

    print("\n=== Summary table ===", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("\n✅ DONE", flush=True)

if __name__ == "__main__":
    main()
