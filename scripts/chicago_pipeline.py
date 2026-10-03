"""
Run the SAME pipeline (DBSCAN + ST-DBSCAN + KDE) on Chicago crime data.
Shows methodology generalizes to real-world event data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

INPUT     = "data/chicago_crime_sample.csv"
OUT_CSV   = "data/chicago_crime_clustered.csv"
FIG_DB    = "figures/fig_chicago_dbscan.png"
FIG_ST    = "figures/fig_chicago_stdbscan.png"
FIG_KDE   = "figures/fig_chicago_kde.png"


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["latitude", "longitude", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # keep Chicago bounding box to remove obvious outliers
    df = df[
        (df["latitude"].between(41.6, 42.05)) &
        (df["longitude"].between(-87.95, -87.5))
    ].copy()

    df["hour"] = df["date"].dt.hour
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    print(f"Cleaned: {len(df)} rows")
    return df


def project_to_meters(df):
    lat0, lon0 = df["lat"].mean(), df["lon"].mean()
    df["x"] = (df["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0))
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


def dbscan_spatial(df, eps=500, min_pts=30):
    """For city-scale data, eps must be in hundreds of meters."""
    labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(df[["x","y"]])
    return labels


def st_dbscan(df, eps_spatial=500, eps_time=3, min_pts=30):
    scale = eps_spatial / eps_time
    feats = np.column_stack([df["x"], df["y"], df["hour"]*scale])
    labels = DBSCAN(eps=eps_spatial, min_samples=min_pts).fit_predict(feats)
    return labels


def compute_metrics(df, labels, tag):
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1
    sil = None
    if len(set(labels[mask])) > 1:
        sil = round(silhouette_score(df.loc[mask, ["x","y"]], labels[mask]), 3)
    print(f"\n📊 {tag}")
    print(f"  clusters : {n_clusters}")
    print(f"  noise    : {n_noise} ({100*n_noise/n_total:.1f}%)")
    print(f"  silhouette: {sil}")
    return {"tag": tag, "n_clusters": n_clusters,
            "n_noise": n_noise, "silhouette": sil}


def plot_clusters(df, labels, title, out_path, max_show=8):
    plt.figure(figsize=(10, 8))
    df = df.copy()
    df["c"] = labels
    noise = df[df["c"] == -1]
    plt.scatter(noise["x"], noise["y"], s=3, c="lightgray",
                label=f"Noise ({len(noise)})", alpha=0.4)
    cmap = plt.get_cmap("tab10")
    for i, cid in enumerate(sorted(set(labels) - {-1})):
        if i >= max_show:
            break
        sub = df[df["c"] == cid]
        if len(sub) < 20:
            continue
        plt.scatter(sub["x"], sub["y"], s=4, color=cmap(i % 10),
                    label=f"C{cid} ({len(sub)})", alpha=0.7)
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(markerscale=3, fontsize=8, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")


def kde_plot(df, out_path):
    from scipy.stats import gaussian_kde
    x, y = df["x"].values, df["y"].values
    pad = 1000
    xx, yy = np.mgrid[x.min()-pad:x.max()+pad:200j,
                      y.min()-pad:y.max()+pad:200j]
    kernel = gaussian_kde(np.vstack([x, y]))
    zz = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, zz, levels=20, cmap="hot")
    plt.title("Chicago crime — KDE density")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs("figures", exist_ok=True)
    df = load_and_clean(INPUT)
    df = project_to_meters(df)

    # spatial DBSCAN — city-scale eps
    labels_s = dbscan_spatial(df, eps=1200, min_pts=50)
    compute_metrics(df, labels_s, "Chicago DBSCAN (spatial)")
    plot_clusters(df, labels_s, "Chicago crime — spatial DBSCAN",
                  FIG_DB)

    # ST-DBSCAN
    labels_st = st_dbscan(df, eps_spatial=1200, eps_time=6, min_pts=50)
    compute_metrics(df, labels_st, "Chicago ST-DBSCAN (spatio-temporal)")
    plot_clusters(df, labels_st, "Chicago crime — ST-DBSCAN",
                  FIG_ST)

    # KDE
    kde_plot(df, FIG_KDE)

    df["cluster_spatial"] = labels_s
    df["cluster_st"]      = labels_st
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()