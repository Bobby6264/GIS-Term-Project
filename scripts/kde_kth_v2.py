"""KDE for KTH with log-scale colorbar so secondary hotspots become visible."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

INPUT = "data/kth_aggregated.csv"
SLICES = [
    ("Morning (6-12)", 6, 12),
    ("Afternoon (12-17)", 12, 17),
    ("Evening (17-22)", 17, 22),
    ("Night (22-6)", 22, 6),
]

def weighted_kde(x, y, w, n=120, pad=200):
    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    xx, yy = np.mgrid[xmin:xmax:n*1j, ymin:ymax:n*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    idx = np.random.choice(len(x), size=min(20_000, int(w.sum())), p=w/w.sum())
    kernel = gaussian_kde(np.vstack([x[idx], y[idx]]))
    zz = kernel(positions).reshape(xx.shape)
    return xx, yy, zz

def plot_log(ax, x, y, w, title):
    xx, yy, zz = weighted_kde(x, y, w)
    # log scale for visibility of secondary hotspots
    zz_log = np.log10(zz + 1e-30)
    cf = ax.contourf(xx, yy, zz_log, levels=25, cmap="hot")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    return cf

def main():
    df = pd.read_csv(INPUT)
    # zoom to main campus where all the activity is
    df = df[(df["x"].between(20000, 23000)) & (df["y"].between(31000, 35000))]
    print(f"Zoomeed to main campus: {len(df):,} rows")

    fig, ax = plt.subplots(figsize=(10, 9))
    cf = plot_log(ax, df["x"].values, df["y"].values, df["count"].values,
                  "KTH campus WiFi — KDE density (all hours, log scale)")
    plt.colorbar(cf, ax=ax, label="log10(Density)")
    plt.tight_layout()
    plt.savefig("figures/fig_kth_kde_allday_log.png", dpi=180)
    plt.close()
    print("Saved figures/fig_kth_kde_allday_log.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    for i, (label, h0, h1) in enumerate(SLICES):
        if h0 < h1:
            sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        else:
            sub = df[(df["hour"] >= h0) | (df["hour"] < h1)]
        if len(sub) < 20:
            continue
        cf = plot_log(axes[i], sub["x"].values, sub["y"].values,
                      sub["count"].values,
                      f"{label}  (n_ap_hours={len(sub):,})")
        plt.colorbar(cf, ax=axes[i], fraction=0.046)

    plt.suptitle("KTH — hotspot evolution across day (log scale)", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/fig_kth_kde_time_slices_log.png", dpi=180)
    plt.close()
    print("Saved figures/fig_kth_kde_time_slices_log.png")

if __name__ == "__main__":
    main()
