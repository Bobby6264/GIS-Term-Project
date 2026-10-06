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
# 1. Parsing (UPDATED to keep Building Name)
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
            "lon": pd.to_numeric(df_rest["Lon"], errors="coerce"),
            "building": df_rest["Building_Name"].astype(str) if "Building_Name" in df_rest.columns else "Unknown"
        }).dropna(subset=["epoch_sec", "lat", "lon"])
        df_rest = df_rest[(df_rest["lat"].between(51.070, 51.090)) & (df_rest["lon"].between(-114.150, -114.110))].copy()

    if not df_path.empty:
        df_path = pd.DataFrame({
            "epoch_sec": pd.to_datetime(df_path["Loct"], format="%d-%b-%Y %I:%M:%S %p", errors="coerce").astype("int64") // 10**9,
            "lat": pd.to_numeric(df_path["Lat"], errors="coerce"),
            "lon": pd.to_numeric(df_path["Lon"], errors="coerce")
        }).dropna(subset=["epoch_sec", "lat", "lon"])
        df_path = df_path[(df_path["lat"].between(51.070, 51.090)) & (df_path["lon"].between(-114.150, -114.110))].copy()

    return df_rest, df_path

# -------------------------------------------------------------
# 2. DBSCAN (UPDATED for Semantic Legends)
# -------------------------------------------------------------
def run_dbscan_and_plot(df, dataset_name, eps_m=12.0, min_pts=60):
    print(f"\n--- Running Spatial DBSCAN on {dataset_name} ---")
    coords = np.radians(df[["lat", "lon"]].values)
    db = DBSCAN(eps=eps_m / EARTH_RADIUS_METERS, min_samples=min_pts, metric="haversine")
    df["cluster"] = db.fit_predict(coords)
    
    n_clusters = len(set(df["cluster"])) - (1 if -1 in df["cluster"].values else 0)
    print(f"Discovered {n_clusters} clusters in {dataset_name}.")

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = df["cluster"] != -1
    
    # Plot Noise
    ax.scatter(df.loc[~mask, "lon"], df.loc[~mask, "lat"], c="lightgrey", s=2, alpha=0.3, label="Noise")
    # Plot Clusters
    scatter = ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], c=df.loc[mask, "cluster"], cmap="tab20", s=10)
    
    ax.set_title(f"Spatial DBSCAN: {dataset_name} ({n_clusters} Clusters)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Dynamic Semantic Legend Below Graph
    unique_clusters = sorted(df.loc[mask, "cluster"].unique())
    handles = [mpatches.Patch(color="lightgrey", label="Noise (-1)")]
    
    for c in unique_clusters[:30]:  # Cap at 30 items for formatting
        color = scatter.cmap(scatter.norm(c))
        
        # Get the most common building name for this geometric cluster
        if "building" in df.columns:
            b_name = df.loc[df["cluster"] == c, "building"].mode()[0]
            # Truncate long building names so the legend doesn't break
            b_name = b_name[:20] + "..." if len(b_name) > 20 else b_name
            label = f"C{c}: {b_name}"
        else:
            label = f"Cluster {c}"
            
        handles.append(mpatches.Patch(color=color, label=label))
        
    if len(unique_clusters) > 30:
        handles.append(mpatches.Patch(color="white", label="... (more omitted)"))

    # Adjust bounding box to make room for the larger semantic legend
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, title="Mapped Building Clusters")
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"1_dbscan_{dataset_name.lower().replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved {out_path}")
    return df
    
# -------------------------------------------------------------
# 2. DBSCAN (UPDATED for High-Contrast Randomized Colors)
# -------------------------------------------------------------
def run_dbscan_and_plot(df, dataset_name, eps_m=12.0, min_pts=60):
    print(f"\n--- Running Spatial DBSCAN on {dataset_name} ---")
    coords = np.radians(df[["lat", "lon"]].values)
    db = DBSCAN(eps=eps_m / EARTH_RADIUS_METERS, min_samples=min_pts, metric="haversine")
    df["cluster"] = db.fit_predict(coords)
    
    mask = df["cluster"] != -1
    unique_clusters = sorted(df.loc[mask, "cluster"].unique())
    n_clusters = len(unique_clusters)
    print(f"Discovered {n_clusters} clusters in {dataset_name}.")

    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 1. Generate a large, highly varied colormap
    color_array = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, max(n_clusters, 1)))
    
    # 2. Shuffle the colors so adjacent buildings get contrasting colors
    np.random.seed(42)
    np.random.shuffle(color_array)
    
    # 3. Map the shuffled colors to specific cluster IDs
    color_dict = {c: color_array[i] for i, c in enumerate(unique_clusters)}
    
    # 4. Apply colors to the dataframe for plotting
    df.loc[mask, "plot_color"] = df.loc[mask, "cluster"].map(color_dict)

    # Plot Noise
    ax.scatter(df.loc[~mask, "lon"], df.loc[~mask, "lat"], c="lightgrey", s=2, alpha=0.3, label="Noise")
    
    # Plot Clusters
    if not df[mask].empty:
        # Convert pandas series of RGB arrays to a standard list for scatter
        ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], c=list(df.loc[mask, "plot_color"]), s=10)
    
    ax.set_title(f"Spatial DBSCAN: {dataset_name} ({n_clusters} Clusters)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Dynamic Semantic Legend Below Graph
    handles = [mpatches.Patch(color="lightgrey", label="Noise (-1)")]
    
    for c in unique_clusters[:30]:  # Cap at 30 items for formatting
        c_color = color_dict[c]
        
        # Get the most common building name for this geometric cluster
        if "building" in df.columns:
            b_name = df.loc[df["cluster"] == c, "building"].mode()[0]
            b_name = b_name[:20] + "..." if len(b_name) > 20 else b_name
            label = f"C{c}: {b_name}"
        else:
            label = f"Cluster {c}"
            
        handles.append(mpatches.Patch(color=c_color, label=label))
        
    if len(unique_clusters) > 30:
        handles.append(mpatches.Patch(color="white", label="... (more omitted)"))

    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, title="Mapped Building Clusters")
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"1_dbscan_{dataset_name.lower().replace(' ', '_')}.png")
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
        # df_rest = run_st_dbscan_and_plot(df_rest, "Rest Points", eps_m=12.0, eps_sec=1800, min_pts=30)
        # run_kde_and_plot(df_rest, "Rest Points")

    # # Run all 3 algorithms independently on the Path dataset
    # if not df_path.empty:
    #     # Relaxing constraints slightly for moving path lines
    #     df_path = run_dbscan_and_plot(df_path, "Path Data", eps_m=15.0, min_pts=40)
    #     df_path = run_st_dbscan_and_plot(df_path, "Path Data", eps_m=15.0, eps_sec=1800, min_pts=20)
    #     run_kde_and_plot(df_path, "Path Data")