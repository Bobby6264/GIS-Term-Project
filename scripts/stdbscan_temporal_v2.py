"""Cleaner temporal profile of ST-DBSCAN clusters."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/kth_aggregated_stdbscan.csv")
clusters = [c for c in sorted(df["cluster"].unique()) if c != -1]

# Build matrix: rows=clusters, cols=hours, values=count
mat = np.zeros((len(clusters), 24))
for i, cid in enumerate(clusters):
    sub = df[df["cluster"] == cid]
    for _, row in sub.iterrows():
        mat[i, int(row["hour"])] += row["count"]

# Normalize per cluster (proportion of daily activity)
mat_norm = mat / mat.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(mat_norm, aspect="auto", cmap="YlOrRd", origin="lower")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels(range(0, 24, 2))
ax.set_yticks(range(len(clusters)))
ax.set_yticklabels([f"C{c}" for c in clusters], fontsize=7)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Cluster")
ax.set_title("KTH ST-DBSCAN — temporal profile per cluster\n(darker = more activity at that hour)")
plt.colorbar(im, ax=ax, label="Proportion of daily activity")
plt.tight_layout()
plt.savefig("figures/fig_kth_stdbscan_temporal_heatmap.png", dpi=180)
plt.close()
print("Saved figures/fig_kth_stdbscan_temporal_heatmap.png")
