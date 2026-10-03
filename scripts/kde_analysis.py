"""
Kernel Density Estimation (KDE) on campus events
------------------------------------------------
Produces:
  1. fig_kde_allday.png     — one heatmap for the full dataset
  2. fig_kde_time_slices.png — 4-panel grid: Morning/Afternoon/Evening/Night
  3. fig_kde_comparison.png  — KDE vs DBSCAN clusters overlay (for slides)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

INPUT_PATH = "data/campus_events.csv"
OUT_DIR    = "figures"

# time slices
SLICES = [
    ("Morning",   6, 12),
    ("Afternoon", 12, 17),
    ("Evening",   17, 22),
    ("Night",     22, 6),   # wraps around midnight
]


# ---------- projection (same as before) ----------
def project_to_meters(df):
    lat0, lon0 = df["lat"].mean(), df["lon"].mean()
    df["x"] = (df["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0))
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


# ---------- KDE helpers ----------
def kde_grid(x, y, n=120, pad=50):
    """Compute KDE over a padded grid. Returns X, Y, Z."""
    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    xx, yy = np.mgrid[xmin:xmax:n*1j, ymin:ymax:n*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(np.vstack([x, y]))
    zz = kernel(positions).reshape(xx.shape)
    return xx, yy, zz


def plot_kde_on_ax(ax, x, y, title, cmap="hot", show_points=False):
    xx, yy, zz = kde_grid(x, y)
    cf = ax.contourf(xx, yy, zz, levels=20, cmap=cmap)
    ax.contour(xx, yy, zz, levels=10, colors="white",
               linewidths=0.4, alpha=0.6)
    if show_points:
        ax.scatter(x, y, s=2, c="cyan", alpha=0.25, edgecolors="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    return cf


# ---------- figure 1: full day ----------
def fig_allday(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    cf = plot_kde_on_ax(ax, df["x"].values, df["y"].values,
                        "KDE Density — All events (full dataset)",
                        show_points=True)
    plt.colorbar(cf, ax=ax, label="Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ---------- figure 2: time slices ----------
def fig_time_slices(df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for i, (label, h0, h1) in enumerate(SLICES):
        if h0 < h1:
            sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        else:  # wraps midnight
            sub = df[(df["hour"] >= h0) | (df["hour"] < h1)]

        ax = axes[i]
        if len(sub) > 20:
            cf = plot_kde_on_ax(
                ax, sub["x"].values, sub["y"].values,
                f"{label}  ({h0:02d}:00–{h1:02d}:00,  n={len(sub)})",
                show_points=False,
            )
            plt.colorbar(cf, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, f"{label}\nnot enough data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)

    plt.suptitle("KDE Hotspot Evolution Across Time Slices",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ---------- figure 3: KDE + DBSCAN overlay ----------
def fig_kde_vs_dbscan(df_events, dbscan_csv, out_path):
    """
    Left  : KDE heatmap
    Right : DBSCAN clusters (from campus_events_dbscan.csv)
    """
    df_db = pd.read_csv(dbscan_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # left: KDE
    ax = axes[0]
    cf = plot_kde_on_ax(ax, df_events["x"].values, df_events["y"].values,
                        "KDE density surface")
    plt.colorbar(cf, ax=ax, fraction=0.046)

    # right: DBSCAN
    ax = axes[1]
    noise = df_db[df_db["cluster"] == -1]
    ax.scatter(noise["x"], noise["y"], s=6, c="lightgray",
               label=f"Noise ({len(noise)})", alpha=0.6)
    cmap = plt.get_cmap("tab10")
    for i, cid in enumerate(sorted(df_db["cluster"].unique())):
        if cid == -1:
            continue
        sub = df_db[df_db["cluster"] == cid]
        ax.scatter(sub["x"], sub["y"], s=8, color=cmap(i % 10),
                   label=f"C{cid} ({len(sub)})", alpha=0.75)
    ax.set_title("DBSCAN clusters (hard boundaries)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# ---------- main ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = project_to_meters(df)
    print(f"Loaded {len(df)} events")

    fig_allday(df, f"{OUT_DIR}/fig_kde_allday.png")
    fig_time_slices(df, f"{OUT_DIR}/fig_kde_time_slices.png")
    fig_kde_vs_dbscan(df, "data/campus_events_dbscan.csv",
                      f"{OUT_DIR}/fig_kde_vs_dbscan.png")


if __name__ == "__main__":
    main()