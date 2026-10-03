"""Parameter sweep for Chicago to pick eps + minPts."""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/chicago_crime_sample.csv")
df = df.dropna(subset=["latitude","longitude","date"]).copy()
df = df[(df["latitude"].between(41.6,42.05)) &
        (df["longitude"].between(-87.95,-87.5))]
lat0, lon0 = df["latitude"].mean(), df["longitude"].mean()
df["x"] = (df["longitude"]-lon0)*111_320*np.cos(np.radians(lat0))
df["y"] = (df["latitude"]-lat0)*110_540
X = df[["x","y"]].values

print(f"n = {len(X)}")
print(f"extent x = {np.ptp(X[:,0]):.0f} m, y = {np.ptp(X[:,1]):.0f} m\n")

for eps in [300, 500, 800, 1200, 1800, 2500, 3500]:
    for min_pts in [20, 50, 100]:
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        n_n = int((labels==-1).sum())
        sil = None
        mask = labels != -1
        if len(set(labels[mask])) > 1 and mask.sum() > 100:
            sil = round(silhouette_score(X[mask], labels[mask]), 3)
        print(f"eps={eps:>5}  minPts={min_pts:>3}  clusters={n_c:>3}  "
              f"noise={n_n:>6} ({100*n_n/len(labels):>5.1f}%)  sil={sil}")
    print()