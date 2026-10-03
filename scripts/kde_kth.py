"""KDE heatmaps for KTH campus WiFi."""
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

def weighted_kde(x, y, w, n=100, pad=200):
    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    xx, yy = np.mgrid[xmin:xmax:n*1j, ymin:ymax:n*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    # resample points according to weight
    idx = np.random.choice(len(x), size=min(30_000, int(w.sum())), p=w/w.sum())
    kernel = gaussian_kde(np.vstack([x[idx], y[idx]]))
    zz = kernel(positions).reshape(xx.shape)
    return xx, yy, zz

def plot_on_ax(ax, x, y, w, title, cmap="hot"):
    xx, yy, zz = weighted_kde(x, y, w)
    cf = ax.contourf(xx, yy, zz, levels=20, cmap=cmap)
    ax.contour(xx, yy, zz, levels=10, colors="white", linewidths=0.4, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    return cf

def main():
    os.makedirs("figures", exist_ok=True)
    print(f"Loading {INPUT} ...", flush=True)
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} rows", flush=True)

    # all-day
    print("All-day KDE...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    cf = plot_on_ax(ax, df["x"].values, df["y"].values, df["count"].values,
                    "KTH campus WiFi — KDE density (all hours)")
    plt.colorbar(cf, ax=ax, label="Density")
    plt.tight_layout()
    plt.savefig("figures/fig_kth_kde_allday.png", dpi=180)
    plt.close()
    print("  saved fig_kth_kde_allday.png", flush=True)

    # time slices
    print("Time-slice KDEs...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    axes = axes.ravel()
    for i, (label, h0, h1) in enumerate(SLICES):
        if h0 < h1:
            sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        else:
            sub = df[(df["hour"] >= h0) | (df["hour"] < h1)]
        ax = axes[i]
        cf = plot_on_ax(ax, sub["x"].values, sub["y"].values, sub["count"].values,
                        f"{label}  (n_ap_hours={len(sub):,})")
        plt.colorbar(cf, ax=ax, fraction=0.046)
    plt.suptitle("KTH campus WiFi — hotspot intensity across day", fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig("figures/fig_kth_kde_time_slices.png", dpi=180)
    plt.close()
    print("  saved fig_kth_kde_time_slices.png", flush=True)

    print("\n✅ DONE", flush=True)

if __name__ == "__main__":
    main()
