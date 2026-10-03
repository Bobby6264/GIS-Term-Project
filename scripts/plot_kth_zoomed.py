"""Re-plot KTH DBSCAN clusters with zoom + jitter so overlaps are visible."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/kth_aggregated_dbscan.csv")

# Keep only the main campus region (removes outlier APs that squash the plot)
mask = (df["x"].between(21000, 22500)) & (df["y"].between(31500, 34000))
df_main = df[mask].copy()
print(f"Main-campus points: {len(df_main):,} of {len(df):,}")

# Jitter coordinates so overlapping points become visible
rng = np.random.default_rng(42)
df_main["x_j"] = df_main["x"] + rng.normal(0, 8, len(df_main))
df_main["y_j"] = df_main["y"] + rng.normal(0, 8, len(df_main))

# Assign consistent color by cluster size rank
sizes = df_main["cluster"].value_counts()
order = sizes.index.tolist()
cmap = plt.get_cmap("tab20")

plt.figure(figsize=(11, 10))
for i, cid in enumerate(order):
    if cid == -1:
        continue
    sub = df_main[df_main["cluster"] == cid]
    plt.scatter(sub["x_j"], sub["y_j"], s=25, alpha=0.75,
                color=cmap(i % 20), label=f"C{cid} ({len(sub):,})")

# annotate the largest clusters with their IDs
for cid in order[:10]:
    if cid == -1: continue
    sub = df_main[df_main["cluster"] == cid]
    cx, cy = sub["x_j"].mean(), sub["y_j"].mean()
    plt.text(cx, cy, str(cid), fontsize=8, fontweight="bold",
             ha="center", va="center", color="black")

plt.title("KTH campus WiFi — DBSCAN clusters (zoomed to main campus, jittered)")
plt.xlabel("X (m)"); plt.ylabel("Y (m)")
plt.legend(markerscale=2, fontsize=6, ncol=3, loc="best", framealpha=0.9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig_kth_dbscan_clusters_zoomed.png", dpi=200)
plt.close()
print("Saved figures/fig_kth_dbscan_clusters_zoomed.png")
