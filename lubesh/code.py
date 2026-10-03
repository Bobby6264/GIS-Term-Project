import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
RAW_DATA_DIR = os.path.expanduser("/home/du2/22CS30034/karan/GIS/lubesh")
PROJECT_DIR = os.path.expanduser("/home/du2/22CS30034/karan/GIS/lubesh")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
EARTH_RADIUS_METERS = 6371008.8

# -------------------------------------------------------------
# 1. Parsing
# -------------------------------------------------------------
def parse_data():
    rest_files = glob.glob(os.path.join(RAW_DATA_DIR, "*RestPoints*.tab"))
    path_files = glob.glob(os.path.join(RAW_DATA_DIR, "*Paths*.tab"))
    
    df_rest = pd.read_csv(rest_files[0], sep="\t", low_memory=False) if rest_files else pd.DataFrame()
    df_path = pd.read_csv(path_files[0], sep="\t", low_memory=False) if path_files else pd.DataFrame()

    if not df_rest.empty:
        df_rest = pd.DataFrame({
            "epoch_sec": pd.to_datetime(df_rest["Loct_Start"], format="%d-%b-%Y %I:%M:%S %p", errors="coerce").astype("int64") // 10**9,
            "lat": pd.to_numeric(df_rest["Lat"], errors="coerce"),
            "lon": pd.to_numeric(df_rest["Lon"], errors="coerce")
        }).dropna()
        df_rest = df_rest[(df_rest["lat"].between(51.070, 51.090)) & (df_rest["lon"].between(-114.150, -114.110))].copy()

    if not df_path.empty:
        df_path = pd.DataFrame({
            "epoch_sec": pd.to_datetime(df_path["Loct"], format="%d-%b-%Y %I:%M:%S %p", errors="coerce").astype("int64") // 10**9,
            "lat": pd.to_numeric(df_path["Lat"], errors="coerce"),
            "lon": pd.to_numeric(df_path["Lon"], errors="coerce")
        }).dropna()
        df_path = df_path[(df_path["lat"].between(51.070, 51.090)) & (df_path["lon"].between(-114.150, -114.110))].copy()

    return df_rest, df_path

# -------------------------------------------------------------
# 2. DBSCAN (Calculates & Plots)
# -------------------------------------------------------------
def run_dbscan_and_plot(df, dataset_name, eps_m=12.0, min_pts=60):
    print(f"\n--- Running Spatial DBSCAN on {dataset_name} ---")
    coords = np.radians(df[["lat", "lon"]].values)
    db = DBSCAN(eps=eps_m / EARTH_RADIUS_METERS, min_samples=min_pts, metric="haversine")
    df["cluster"] = db.fit_predict(coords)
    
    n_clusters = len(set(df["cluster"])) - (1 if -1 in df["cluster"].values else 0)
    print(f"Discovered {n_clusters} clusters in {dataset_name}.")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = df["cluster"] != -1
    
    # Plot Noise
    ax.scatter(df.loc[~mask, "lon"], df.loc[~mask, "lat"], c="lightgrey", s=2, alpha=0.3, label="Noise")
    # Plot Clusters
    scatter = ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], c=df.loc[mask, "cluster"], cmap="tab20", s=10)
    
    ax.set_title(f"Spatial DBSCAN: {dataset_name} ({n_clusters} Clusters)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Dynamic Legend Below Graph
    unique_clusters = sorted(df.loc[mask, "cluster"].unique())
    handles = [mpatches.Patch(color="lightgrey", label="Noise (-1)")]
    for c in unique_clusters[:30]:  # Cap at 30 items for formatting
        color = scatter.cmap(scatter.norm(c))
        handles.append(mpatches.Patch(color=color, label=f"Cluster {c}"))
    if len(unique_clusters) > 30:
        handles.append(mpatches.Patch(color="white", label="... (more omitted)"))

    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=8, title="Color Mapping")
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"1_dbscan_{dataset_name.lower().replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved {out_path}")
    return df

# -------------------------------------------------------------
# 3. ST-DBSCAN (Calculates & Plots)
# -------------------------------------------------------------
def run_st_dbscan_and_plot(df, dataset_name, eps_m=12.0, eps_sec=1800, min_pts=30):
    print(f"\n--- Running ST-DBSCAN on {dataset_name} ---")
    coords = np.radians(df[["lat", "lon"]].values)
    times = df["epoch_sec"].values
    n = len(df)
    
    tree = BallTree(coords, metric="haversine")
    spatial_neighbors = tree.query_radius(coords, r=eps_m / EARTH_RADIUS_METERS)
    
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    progress_interval = max(1, n // 10)
    for i in range(n):
        if i % progress_interval == 0 and i > 0:
            print(f"    Progress: {int((i / n) * 100)}%...")
            
        if visited[i]: continue
        visited[i] = True
        
        cand = spatial_neighbors[i]
        st_neighbors = cand[np.abs(times[cand] - times[i]) <= eps_sec]
        
        if len(st_neighbors) >= min_pts:
            cluster_id += 1
            labels[i] = cluster_id
            seed_set = list(st_neighbors[st_neighbors != i])
            
            while seed_set:
                curr = seed_set.pop(0)
                if not visited[curr]:
                    visited[curr] = True
                    c_cand = spatial_neighbors[curr]
                    c_st = c_cand[np.abs(times[c_cand] - times[curr]) <= eps_sec]
                    if len(c_st) >= min_pts:
                        seed_set.extend([p for p in c_st if p not in seed_set and not visited[p]])
                if labels[curr] == -1:
                    labels[curr] = cluster_id
                    
    df["st_cluster"] = labels
    n_st_clusters = cluster_id
    print(f"Discovered {n_st_clusters} spatio-temporal clusters in {dataset_name}.")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask_st = df["st_cluster"] != -1
    
    ax.scatter(df.loc[~mask_st, "lon"], df.loc[~mask_st, "lat"], c="lightgrey", s=2, alpha=0.3)
    scatter_st = ax.scatter(df.loc[mask_st, "lon"], df.loc[mask_st, "lat"], c=df.loc[mask_st, "st_cluster"], cmap="turbo", s=10)
    
    ax.set_title(f"ST-DBSCAN: {dataset_name} ({n_st_clusters} Events)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    unique_st = sorted(df.loc[mask_st, "st_cluster"].unique())
    handles_st = [mpatches.Patch(color="lightgrey", label="Noise (-1)")]
    for c in unique_st[:30]:
        color = scatter_st.cmap(scatter_st.norm(c))
        handles_st.append(mpatches.Patch(color=color, label=f"Time Event {c}"))
    if len(unique_st) > 30:
        handles_st.append(mpatches.Patch(color="white", label="... (more omitted)"))

    ax.legend(handles=handles_st, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=8, title="Color Mapping")
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"2_st_dbscan_{dataset_name.lower().replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved {out_path}")
    return df

# -------------------------------------------------------------
# 4. KDE (Calculates & Plots)
# -------------------------------------------------------------
def run_kde_and_plot(df, dataset_name):
    print(f"\n--- Running KDE on {dataset_name} ---")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_df = df.sample(min(20000, len(df)), random_state=42) if len(df) > 20000 else df
    x, y = plot_df["lon"].values, plot_df["lat"].values
    kde = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():150j, y.min():y.max():150j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    contour = ax.contourf(xi, yi, zi, levels=20, cmap="inferno")
    ax.set_title(f"KDE Density: {dataset_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"3_kde_{dataset_name.lower().replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[*] Saved {out_path}")

# -------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------
if __name__ == "__main__":
    df_rest, df_path = parse_data()
    
    # Run all 3 algorithms independently on the Rest Points dataset
    if not df_rest.empty:
        df_rest = run_dbscan_and_plot(df_rest, "Rest Points", eps_m=12.0, min_pts=60)
        df_rest = run_st_dbscan_and_plot(df_rest, "Rest Points", eps_m=12.0, eps_sec=1800, min_pts=30)
        run_kde_and_plot(df_rest, "Rest Points")

    # Run all 3 algorithms independently on the Path dataset
    if not df_path.empty:
        # Relaxing constraints slightly for moving path lines
        df_path = run_dbscan_and_plot(df_path, "Path Data", eps_m=15.0, min_pts=40)
        df_path = run_st_dbscan_and_plot(df_path, "Path Data", eps_m=15.0, eps_sec=1800, min_pts=20)
        run_kde_and_plot(df_path, "Path Data")