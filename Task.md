# Spatio-Temporal Hotspot Analysis of Campus Events
## Using DBSCAN, ST-DBSCAN and KDE

**Course:** Geographical Information Systems
**Project:** Spatio-Temporal Hotspot Analysis of Campus Events

---

## TABLE OF CONTENTS

1. Project Requirement
2. Datasets Used
3. Synthetic Campus Dataset
4. KTH/Campus WiFi Dataset
5. Algorithms Used
6. Concept Reference (silhouette, AP, aggregation, eps, etc.)
7. Scripts and What They Generate
8. KTH Results — Figure-by-Figure
9. Results Summary Tables
10. Interpretation
11. Limitations
12. Presentation Plan
13. Q&A Prep
14. Project Structure
15. How to Reproduce

---

## 1. PROJECT REQUIREMENT

The course project requires a **spatio-temporal hotspot analysis**
of **campus events** using:

- **DBSCAN** — Density-Based Spatial Clustering of Applications with Noise
- **ST-DBSCAN** — Spatio-temporal variant of DBSCAN
- **KDE** — Kernel Density Estimation (heatmap smoothing)

The goal is to identify **where** events concentrate (spatial
hotspots) and **when** they concentrate (temporal patterns), and
combine both into a spatio-temporal view.

No KGP-specific dataset was provided. The TA confirmed that any
publicly available campus event dataset may be used (e.g., campus
WiFi traces from the CRAWDAD archive).

---

## 2. DATASETS USED

Two datasets, each with a distinct role:

| Dataset | Role | Size |
|---------|------|------|
| **Synthetic campus events** | Validation (has ground truth) | 5,000 events |
| **KTH/Campus WiFi** (CRAWDAD) | Primary real-world dataset | 5,589,617 events |

The synthetic data proves the method **works** (we can verify it
recovers known hotspots). The KTH data proves the method
**generalizes** to messy real campus data.

---

## 3. SYNTHETIC CAMPUS DATASET

### 3.1 What it is
A synthetic generator creates **5,000 events** across **3 known
hotspot locations** over **30 simulated days**:

- **Library** — 1,500 events, peaks at 10–11am and 2–4pm
- **Cafeteria** — 1,800 events, peaks at 12–1pm and 7–8pm
- **Sports Complex** — 1,200 events, peaks at 6–7am and 5–6pm
- **500 noise events** — uniformly scattered, no pattern

### 3.2 Why synthetic
Real data has no ground truth — we cannot quantitatively verify
whether an algorithm found "the right" hotspots. Synthetic data has
known hotspots, so we can measure correctness. This is standard
practice for validating clustering algorithms.

### 3.3 Columns in `data/campus_events.csv`

| Column | Meaning |
|--------|---------|
| `event_id` | Unique event ID |
| `lon`, `lat` | Longitude, latitude |
| `day` | Simulated day (0–29) |
| `hour` | Hour of day (0–23) |
| `type` | Hotspot name (Library/Cafeteria/Sports/other) |

### 3.4 Code that generates it
- **Script:** `scripts/generate_data.py`
- **Output:** `data/campus_events.csv`

### 3.5 Results on synthetic data

| Method | Clusters | Noise % | Silhouette |
|--------|----------|---------|------------|
| DBSCAN (spatial) | **3** | 8.8% | **0.901** |
| ST-DBSCAN | **6** | 9.4% | **0.548** |

**Interpretation:**
- Spatial DBSCAN recovered the 3 planted hotspots exactly
- ST-DBSCAN found 6 clusters — each hotspot split into 2 temporal
  regimes (Cafeteria lunch vs dinner; Sports morning vs evening)
- Noise count (~440) matches the 500 planted noise events
- **Key takeaway:** adding the temporal dimension reveals structure
  invisible to spatial-only analysis

---

## 4. KTH/CAMPUS WiFi DATASET (PRIMARY REAL DATA)

### 4.1 What the dataset is
**CRAWDAD KTH/Campus** — real wireless network measurements from
**KTH Royal Institute of Technology**, Stockholm, Sweden.

- **Collected:** January 2014 – December 2015
- **Location:** Two large campuses + four small campuses
- **Contents:** Records of authenticated user associations to WiFi
  access points, plus AP location mappings collected via war-walking
- **Source:** https://ieee-dataport.org/open-access/crawdad-kthcampus
- **Citation:** Ljubica Pajevic, Gunnar Karlsson, Viktoria Fodor,
  kth/campus, DOI 10.15783/c7-5r6x-4b46

### 4.2 Raw data files

| File | Contents |
|------|----------|
| `data/kth/2014_01.csv.gz` | January 2014 WiFi association log (113 MB compressed, ~5.6M rows) |
| `data/kth/APlocations.txt` | AP name → x, y, floor coordinate mapping |

### 4.3 Raw columns

**`2014_01.csv.gz`:**

| Column | Meaning |
|--------|---------|
| `timestamp` | When the device connected |
| `client` | Anonymized MAC address (device ID) |
| `AP` | Access point name (e.g., `Bldg11AP21`) |

**`APlocations.txt`:**

| Column | Meaning |
|--------|---------|
| `AP` | Access point name |
| `x_coordinate(m)` | X coordinate in meters (local KTH system) |
| `y_coordinate(m)` | Y coordinate in meters |
| `floor` | Building floor |

### 4.4 Which portion we used
**January 2014 only** — one month, 5.6M events. Enough to
characterize a month of activity. Larger samples don't change the
methodological story.

### 4.5 Data cleaning and feature extraction
- Dropped events whose `AP` has no entry in `APlocations.txt`
  (2,482 of 5.6M — negligible)
- Merged events with AP coordinates (inner join)
- Extracted `hour` from timestamp
- Dropped duplicate (timestamp, AP) rows — 43,851 duplicates
- **Result:** `data/kth_events.csv` — 5,589,617 events

### 4.6 Aggregation into spatio-temporal buckets
5.6M raw events is too large for interactive clustering and contains
many events at **identical (x, y)** coordinates (multiple APs in a
building share coordinates). We aggregated events into
**(AP, hour, day) counts**:

- **Result:** `data/kth_aggregated.csv` — **346,770 rows**
- Mean event count per bucket: **16**
- Max: **923 events** in a single (AP, hour, day) — a clear hotspot
  signature

This is the **standard GIS workflow**: aggregate raw point events
into a density surface before clustering.

### 4.7 Columns in `data/kth_aggregated.csv`

| Column | Meaning |
|--------|---------|
| `AP` | Access point name |
| `x`, `y` | Coordinates in meters |
| `hour` | Hour of day (0–23) |
| `day` | Day of month (1–31) |
| `count` | Number of WiFi associations at this (AP, hour, day) |

### 4.8 Collapse across days (for ST-DBSCAN)
Because we want **daily patterns** (not day-to-day variation), we
collapse `day` by summing counts, giving one row per **(AP, hour)**:

- **Result:** 20,129 rows
- This makes time a cyclic daily feature (hour-of-day 0–23)

---

## 5. ALGORITHMS USED

### 5.1 DBSCAN
Density-based clustering. Groups points that are densely packed
(within radius `eps`, with at least `minPts` neighbors). Points in
low-density regions are labelled **noise** (−1).

**Chosen over K-means because:**
- No need to specify `k`
- Finds arbitrary cluster shapes
- Explicitly handles noise

### 5.2 ST-DBSCAN
Same as DBSCAN but requires closeness in **both space AND time**.
Implementation: combine (x, y, hour × scale) into one feature space
and run DBSCAN. The scale factor maps `eps_time` hours to
`eps_spatial` meters so one `eps` works across all axes.

### 5.3 KDE
Smooths points into a continuous density surface (heatmap) using
Gaussian kernels. Complementary to DBSCAN:
- DBSCAN → discrete clusters with hard boundaries
- KDE → continuous intensity surface

For KTH, we use event `count` as weights so busier AP-hours
contribute more.

---

## 6. CONCEPT REFERENCE

### 6.1 What is Silhouette?

**Silhouette** is a **numerical score between −1 and +1** that
measures how well each point fits its own cluster compared to the
nearest other cluster.

**How it's computed:** For each point i:
- **a(i)** = mean distance to all other points in its own cluster
- **b(i)** = mean distance to all points in the nearest other cluster
- **silhouette(i) = (b(i) − a(i)) / max(a(i), b(i))**

The overall score is the mean over all points.

**Interpretation:**

| Silhouette | Meaning |
|------------|---------|
| Close to +1 | Point is well inside its own cluster, far from others |
| Around 0 | Point lies on the boundary between two clusters |
| Negative | Point might belong to a different cluster |

**Why we use it:**
- Objective measure of cluster quality
- Lets us compare algorithms on the same data
- Lets us compare datasets
- Gives the panel a number to react to

**Caveats:**
- Silhouette ignores noise labels (−1)
- On very dense datasets, silhouette can be misleadingly high
- Real data usually gives lower silhouette than synthetic

**Our silhouette results explained:**

| Dataset | Method | Silhouette | Why |
|---------|--------|------------|-----|
| Synthetic | DBSCAN | **0.901** | Gaussian blobs are well-separated by construction |
| Synthetic | ST-DBSCAN | **0.548** | Time dimension makes clusters more crowded |
| KTH | DBSCAN | **0.997** | APs in same building share coordinates — clusters are physically perfect |
| KTH | ST-DBSCAN | **0.246** | Time dimension added — clusters overlap in time |
| KTH | Time-sliced | **0.997** | Same as spatial because slices share coordinates |

**Takeaway:** Silhouette measures *geometric compactness*, not
*meaningfulness*. KTH's 0.997 doesn't mean KTH clusters are more
"correct" — it means KTH clusters are physically very tight.

---

### 6.2 What is an AP (Access Point)?

**AP = Access Point.** A physical WiFi router/antenna installed in
a building that client devices connect to.

**In our KTH dataset:**
- KTH has **1,123 access points** across its campuses
- Each AP has a name like `Bldg11AP21` (building 11, AP 21)
- Each AP has coordinates `(x, y, floor)` in `APlocations.txt`
- Many APs share the **same (x, y)** because multiple APs are
  installed in the same building — on different floors or corners,
  but the coarse x,y in the mapping file is identical

**Why APs matter:** When a device connects to WiFi, the network
logs *which AP* (location) and *when* (time). Each WiFi
association event = a spatio-temporal point.

**Why "many APs share coordinates" is a problem:** If
`Bldg11AP1`, `Bldg11AP2`, `Bldg11AP3` all map to (21534, 32313),
then events at those three APs are spatially indistinguishable.
This is why our k-distance graph is flat (all zeros) — the k
nearest neighbors of any point are often at distance 0.

**What we did about it:** We accept it. Clusters represent
**buildings** (52 of them), not individual APs (1,123 of them).
This is still meaningful — buildings are valid hotspot units.

---

### 6.3 Why and How We Aggregated the CSV Data

**The problem with raw events:** 5,589,617 rows, one per WiFi
association:

```text
timestamp                client      AP
2014-01-01 00:00:24      b7f22f...   Bldg11AP21
2014-01-01 00:00:25      a3d91b...   Bldg11AP21
2014-01-01 00:00:27      c19ff2...   Bldg11AP21
...
```

Issues:
1. Too many rows — slow to cluster (DBSCAN took 125 s)
2. Redundant — 923 events at the same (AP, hour, day) all say the
   same thing: "this place was busy at this time"
3. Cannot compute density — every row has weight 1, but 923 events
   should weigh 923

**What aggregation means:** Collapse multiple rows into one row,
summarizing a measure (count). We grouped by **(AP, hour, day)** and
**counted** events per group.

Result: **5,589,617 rows → 346,770 rows** (16× reduction).

**Why this is the right thing to do:**

| Reason | Explanation |
|--------|-------------|
| Semantic | We care about "how busy was this AP at this hour" — count captures it |
| Computational | 346k rows cluster ~3× faster than 5.6M |
| Robust | Duplicate points no longer dominate k-distance |
| Standard GIS practice | Aggregating point events to density grids is textbook |

**The code:**

```python
agg = (
    df.groupby(["AP", "x", "y", "hour", "day"])
      .size()
      .reset_index(name="count")
)
```

- `groupby([...])` — group rows by these 5 columns
- `.size()` — count rows in each group
- `.reset_index(name="count")` — convert back to DataFrame with a `count` column

- **Script:** `scripts/aggregate_kth.py`
- **Output:** `data/kth_aggregated.csv`

**Aggregation stages in our project:**

| Stage | Rows | What it represents |
|-------|------|--------------------|
| Raw KTH log | 5,589,617 | One row per WiFi association |
| Merged with AP locations | 5,589,617 | Added x, y, floor |
| Aggregated (AP, hour, day) | 346,770 | One row per (AP, hour, day) with count |
| Aggregated (AP, hour) for ST-DBSCAN | 20,129 | Collapsed across days |

---

### 6.4 On What Basis Are We Clustering?

**Features used:**

- **Spatial DBSCAN — 2 features:**
  - `x` — meters east of the campus origin
  - `y` — meters north of the campus origin

- **ST-DBSCAN — 3 features:**
  - `x`, `y` — as above
  - `hour × scale` — hour of day, scaled so `eps_time` hours equals `eps_spatial` meters

- **KDE** — no clustering, but weighted by `count` per (AP, hour).

**Why these features:**

- `x`, `y` are the spatial dimension — clustering finds geographically
  close APs
- `hour` is the temporal dimension — forces clusters to also be close
  in time
- We do **NOT** cluster on `client` (privacy + irrelevant), `day`
  (day-of-month noise), `floor` (incomplete), or `count` (would
  distort distance metric; used as KDE weight instead)

**Distance metric:** Euclidean by default. For 2D:
`sqrt((x1-x2)² + (y1-y2)²)` in meters. For 3D ST-DBSCAN:
`sqrt((x1-x2)² + (y1-y2)² + ((h1-h2)×scale)²)`.

The scale factor (`= eps_spatial / eps_time`) is critical: without
it, the hour axis (range 0–23) would be dwarfed by x, y (range
0–30,000 m). With it, 1 hour = same weight as `scale` meters.

**What "clustering" means physically:** Two events are in the same
cluster if they are within `eps` meters (spatial) AND (for ST-DBSCAN)
within `eps_time` hours. Clusters form by density-connectivity:
if A is near B, and B is near C, then A, B, C are all in the same
cluster — even if A and C are far apart. This is called **chaining**.

**No labels, no training:** DBSCAN is unsupervised — we don't
tell it how many clusters or where they are. Only inputs: the
feature matrix and `eps`, `minPts`.

### 6.5 What is eps? Why Does It Change the Number of Clusters?

**Definition:** `eps` (epsilon) is the radius of the
neighborhood around each point. Two points are "neighbors" if
their distance is ≤ `eps`. It's the most important parameter in
DBSCAN — it directly controls cluster granularity.

**The other parameter:** `minPts` is the minimum number of
neighbors a point needs to be a "core point." Core points seed
clusters. Non-core points within `eps` of a core point become
border points. Everything else is noise (−1).

**How they work together:** A cluster grows:

1. Find a core point (has ≥ `minPts` neighbors within `eps`)
2. Add all its neighbors to the cluster
3. Repeat for each newly added point

So:

- **Small eps** → few points have enough neighbors → many small
  clusters + lots of noise
- **Large eps** → most points have many neighbors → few large
  clusters + little noise
- **Too large** → everything becomes one giant cluster

**Our KTH eps sweep (raw evidence):**

| eps (m) | clusters | noise % | silhouette | Interpretation |
|---------|----------|---------|------------|----------------|
| 20 | 53 | 0.0% | 0.999 | Building-level (each AP group isolated) |
| 30 | 52 | 0.0% | 0.997 | Building-level — **CHOSEN** |
| 50 | 34 | 0.0% | 0.775 | Multiple buildings merged |
| 75 | 19 | 0.0% | 0.441 | District-level merging |
| 100 | 13 | 0.0% | 0.491 | Neighborhood-level |
| 150 | 10 | 0.0% | 0.579 | Too coarse |

**Reading the table:**

- At eps = 20–30 m, clusters = 52–53 — matches KTH building count
- At eps = 50 m, cluster count drops to 34 — adjacent buildings merge
- At eps = 100 m, cluster count is 13 — the algorithm is grouping
  neighborhoods, not buildings
- Silhouette is highest in the 20–30 m range

**Why eps = 30 was chosen:**

- Silhouette is highest (0.997)
- Cluster count (52) matches KTH building count (~50)
- The k-distance graph was flat (0) because of shared AP
  coordinates, so the sweep table replaces it as our justification

**Where eps appears in the code:**

`scripts/dbscan_kth_agg.py`:

```python
EPS_M = 30.0
labels = DBSCAN(eps=EPS_M, min_samples=MIN_PTS, ...)
```

`scripts/stdbscan_kth.py`:

```python
EPS_SPATIAL = 30.0
EPS_TIME    = 2.0
scale = EPS_SPATIAL / EPS_TIME
feats = np.column_stack([
    df["x"].values, df["y"].values, df["hour"].values * scale,
])
labels = DBSCAN(eps=EPS_SPATIAL, min_samples=MIN_PTS, ...)
```

`scripts/time_sliced_dbscan.py`:

```python
EPS_M = 30.0
for label, h0, h1 in SLICES:
    sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
    labels = DBSCAN(eps=EPS_M, min_samples=MIN_PTS, ...)
```

**One-line summary:**

> `eps` is the maximum distance for two points to be neighbors.
> Small eps → many small clusters; large eps → few big
> clusters. We chose eps = 30 m for KTH because it gave the
> highest silhouette (0.997) and a cluster count (52) matching the
> campus's building count.

---

### 6.6 Why Not Use Other Clustering Algorithms?

| Algorithm | Why not for our data |
|-----------|---------------------|
| K-means | Requires specifying k upfront; assumes spherical clusters; no noise handling |
| Hierarchical | O(n²) memory — infeasible on 346k points |
| Mean-shift | Needs bandwidth parameter; doesn't handle noise as cleanly |
| Gaussian Mixture Models | Assumes Gaussian distributions per cluster; KTH APs are not Gaussian |
| HDBSCAN | Overkill for a course project |

DBSCAN is chosen because: no k needed, handles arbitrary
shapes, explicit noise labeling, two intuitive parameters with
physical meaning.

### 6.7 What Is a Hotspot?

A **hotspot** is a concentration of events in space and time.

- **Spatial hotspot** — a location where many events occur.
  Identified by a spatial DBSCAN cluster or a bright KDE region.
- **Spatio-temporal hotspot** — a location during a specific time
  window where events concentrate. Identified by ST-DBSCAN or by a
  bright region in a time-sliced KDE.

**Our hotspots on KTH:**

- 52 spatial hotspots = 52 buildings with concentrated WiFi activity
- Peak: up to 923 events per (AP, hour, day) in the busiest building
- Temporal: busiest hour is 13:00 (23,495 events citywide); quietest
  is 04:00 (3,582 events)

---

### 6.8 What Is a WiFi Association Event?

When a phone or laptop connects to WiFi, the network logs an
association event: "device X connected to AP Y at time T."
When the device disconnects or moves to another AP, another event
is logged.

For KTH, the dataset provides these association events anonymized:
device identity replaced by a hashed MAC, timestamp preserved, AP
name preserved.

Each event = one row = one spatio-temporal point.

---

### 6.9 What Is KDE (Kernel Density Estimation)?

KDE is a smoothing technique that turns discrete points into a
continuous density surface.

**How it works:** For every point, place a small "bump" (Gaussian
kernel). Sum all bumps. The result is a smooth surface where peaks
= high-density areas (hotspots) and valleys = low-density areas.

**Why use KDE alongside DBSCAN:**

| DBSCAN | KDE |
|--------|-----|
| Discrete clusters with hard boundaries | Continuous density surface |
| Tells you where clusters are | Tells you how intense each region is |
| Labels every point | Smooths across boundaries |

They answer different questions and are complementary.

**Weighted KDE for KTH:** Each (AP, hour) row has a `count`. We use
weighted KDE so a bucket with 923 events contributes more than a
bucket with 1 event — the surface reflects event density, not just
distinct-AP density.

---

### 6.10 Why Is KDE Time-Sliced?

Running KDE on the whole dataset gives one density map — where
hotspots are overall.

Running KDE on time slices (Morning / Afternoon / Evening / Night)
gives four maps — where hotspots are at each time.

**Comparing the four maps reveals the temporal story:**

- Are hotspots in the same place all day? → intensity changes,
  location stable
- Do hotspots move? → different locations active at different times

**On KTH:** all four slices show the same dominant hotspot
location; intensity changes dramatically. **Finding:** location is
architecture-determined; intensity is time-determined.

---

### 6.11 Why Time-Sliced DBSCAN Instead of ST-DBSCAN?

We ran both:

- **ST-DBSCAN** (all data, one 3D space): 54 clusters
- **Time-sliced DBSCAN** (5 slices, separate 2D clustering): 51–52
  clusters

**Why do both:**

- ST-DBSCAN models "close in space AND time" as one metric. Finds
  dual-regime buildings.
- Time-sliced DBSCAN treats each time window independently. Finds
  the stable spatial structure that persists across slices.

On KTH, both give similar results because spatial structure
dominates. On synthetic, ST-DBSCAN gives a much richer story
(3 → 6 clusters split by time). Presenting both shows the method's
flexibility — the right tool depends on whether spatial or temporal
structure dominates.

---

### 6.12 Why No Training? "How Is the Model Trained?"

DBSCAN, ST-DBSCAN and KDE are **not trained** — they are
unsupervised, non-parametric algorithms.

| Aspect | Supervised ML | Our methods |
|--------|---------------|-------------|
| Learns parameters? | Yes | No |
| Has weights? | Yes | No |
| Loss function? | Yes | No |
| Train/test split? | Yes | No |
| Overfitting risk? | Yes | No |
| Deterministic output? | No | Yes (given hyperparams) |
| Terminology | "Train", "fit", "predict" | "Run", "apply", "cluster" |

The only choices we make are hyperparameters: `eps`, `minPts`,
`eps_time`. We select them via a parameter sweep and by looking at
the silhouette score.

**Analogy:** DBSCAN is like a sieve with a fixed hole size — you
pour data through it and clusters fall out. There is no learning.
---

## 7. SCRIPTS AND WHAT THEY GENERATE

### 7.1 Synthetic pipeline

| Script | Generates |
|--------|-----------|
| `scripts/generate_data.py` | `data/campus_events.csv` — synthetic events |
| `scripts/tune_eps.py` | `data/eps_sweep.csv` — eps sweep table |
| `scripts/dbscan_spatial.py` | `figures/fig_kdistance.png`, `figures/fig_dbscan_clusters.png`, `data/campus_events_dbscan.csv`, `data/metrics_dbscan.csv` |
| `scripts/stdbscan.py` | `figures/fig_stdbscan_clusters.png`, `data/campus_events_stdbscan.csv`, `data/metrics_stdbscan.csv` |
| `scripts/kde_analysis.py` | `figures/fig_kde_allday.png`, `figures/fig_kde_time_slices.png`, `figures/fig_kde_vs_dbscan.png` |
| `scripts/summarize.py` | `data/results_summary.csv` |

### 7.2 KTH pipeline

| Script | Generates |
|--------|-----------|
| `scripts/inspect_kth.py` | Printed inspection of raw KTH data |
| `scripts/kth_to_events.py` | `data/kth_events.csv` — merged events with coordinates |
| `scripts/aggregate_kth.py` | `data/kth_aggregated.csv` — aggregated (AP, hour, day) counts |
| `scripts/dbscan_kth_agg.py` | `figures/fig_kth_kdistance.png`, `figures/fig_kth_dbscan_clusters.png`, `data/kth_aggregated_dbscan.csv`, `data/metrics_kth_dbscan.csv` |
| `scripts/stdbscan_kth.py` | `figures/fig_kth_stdbscan_clusters.png`, `data/kth_aggregated_stdbscan.csv`, `data/metrics_kth_stdbscan.csv` |
| `scripts/time_sliced_dbscan.py` | `figures/fig_kth_time_sliced.png`, `data/metrics_kth_time_sliced.csv` |
| `scripts/kde_kth.py` | `figures/fig_kth_kde_allday.png`, `figures/fig_kth_kde_time_slices.png` |

---

## 8. KTH RESULTS — FIGURE-BY-FIGURE

### 8.1 `fig_kth_kdistance.png` — k-distance graph

Sorted distance to each point's 3rd nearest neighbor.

**Observation:** all values are 0.00 m — because many APs in the
same building share identical (x, y) coordinates. Not useful for
choosing eps on KTH. eps was chosen via the parameter sweep
instead. Figure kept for completeness; not used in presentation.

### 8.2 `fig_kth_dbscan_clusters.png` — spatial DBSCAN

Scatter plot of all 346,770 aggregated events, colored by cluster.

**Observation:** 52 clusters, each a distinct color. Silhouette
0.997 (near-perfect). Clusters correspond to KTH's ~52 buildings.
A few outlier points (small clusters far from the main campus)
reflect APs in remote buildings.

### 8.3 `fig_kth_stdbscan_clusters.png` — ST-DBSCAN

Two panels:

- **Left:** spatial view of ST-DBSCAN clusters (54 total)
- **Right:** temporal profile — histogram of hours per cluster

**Observation:** 54 clusters vs 52 for spatial-only. The +2 extra
clusters are locations whose activity patterns split into two
time regimes. Temporal histograms show all clusters are active
across all 24 hours, but with varying peak times.

### 8.4 `fig_kth_time_sliced.png` — time-sliced spatial DBSCAN

Five side-by-side panels, one per time window:

Night (0–6), Morning (6–12), Afternoon (12–17), Evening (17–22),
Late (22–24)

**Observation:** every time window yields 51–52 clusters — the
spatial structure is stable across the day. What changes is
event intensity: 25,495 events at night vs 116,636 in the
afternoon. **Key finding:** location of hotspots is
architecture-determined; intensity is time-determined.

### 8.5 `fig_kth_kde_allday.png` — all-day KDE density

A smooth density surface of all WiFi events combined.

**Observation:** one dominant hotspot in the top-right region of
the map (a major building) plus a small secondary hotspot. Most of
the campus has very low WiFi density. Reflects KTH's activity
concentration.

### 8.6 `fig_kth_kde_time_slices.png` — KDE across 4 time windows

Four heatmaps: Morning / Afternoon / Evening / Night.

**Observation:** the same dominant hotspot appears in all four
panels — confirming spatial location is stable. But intensity
varies dramatically: peak at afternoon, low at night. Visual
proof of the temporal signal.

---

## 9. RESULTS SUMMARY TABLES

### 9.1 Synthetic campus data

| Method | Clusters | Noise % | Silhouette |
|--------|----------|---------|------------|
| DBSCAN (spatial) | 3 | 8.8% | 0.901 |
| ST-DBSCAN | 6 | 9.4% | 0.548 |

### 9.2 KTH campus WiFi data

| Method | eps | Clusters | Noise % | Silhouette |
|--------|-----|----------|---------|------------|
| DBSCAN (spatial) | 30 m | 52 | 0.0% | 0.997 |
| ST-DBSCAN | 30 m / 2 hr | 54 | 0.0% | 0.246 |
| Time-sliced DBSCAN (avg) | 30 m | 52 | 0.0% | 0.997 |

**Additional numeric finding:** Hourly WiFi event count peaks at
23,495 (13:00) and drops to 3,582 (04:00) — a 6.5× swing
between midday and pre-dawn.

---

## 10. INTERPRETATION

### 10.1 Synthetic dataset: cleaner story

- 3 → 6 clusters when time is added → temporal dimension reveals
  structure
- High silhouette (0.90) because synthetic Gaussian hotspots are
  well-separated
- Proves the methodology works when ground truth is known

### 10.2 KTH dataset: real-world story is different

- 52 → 54 clusters when time is added → only +2 clusters
- Why? APs in the same building share identical coordinates, so
  spatial structure is already very tight
- Temporal dimension affects intensity, not location
- Silhouette drops (0.997 → 0.246) because 3D space is more crowded

### 10.3 Combined message

Both datasets confirm the pipeline works: DBSCAN reliably
recovers spatial hotspots, ST-DBSCAN adds marginal temporal
splits, and KDE visualizes how activity intensity shifts across
the day. The synthetic case demonstrates clean cluster recovery
when ground truth is known; the KTH case demonstrates the
pipeline on a real campus with noise, coordinate aggregation,
and architecture-driven structure.

---

## 11. LIMITATIONS

- Synthetic data assumes Gaussian clusters — real events don't
  obey this; KTH results show this clearly
- KTH APs share coordinates — many APs in one building map to
  identical (x, y), limiting spatial resolution of clustering
- One month of KTH data used — seasonality across months not
  captured
- Cyclic time not modeled — events at 23:00 and 01:00 are
  treated as far apart even though they are temporally adjacent
- No ground truth on KTH — we can't verify whether the 52
  clusters are "correct"
- Silhouette is geometry-focused — high scores on KTH don't
  imply semantically meaningful clusters

---

## 12. PRESENTATION PLAN (8 SLIDES)

1. **Title** — project name, members, course
2. **Problem & Motivation** — campus event hotspots, why
   spatio-temporal
3. **Datasets** — synthetic (validation) + KTH/Campus WiFi (primary)
4. **Methodology** — DBSCAN + ST-DBSCAN + KDE pipeline diagram
5. **Synthetic results** — 3 clusters recovered exactly (sil 0.90),
   ST-DBSCAN splits to 6 (sil 0.55)
6. **KTH spatial + ST-DBSCAN results** — 52 clusters (sil 0.997);
   ST-DBSCAN: 54 clusters
7. **KTH temporal analysis** — time-sliced DBSCAN (5 panels) + KDE
   evolution (4 panels) — intensity shifts, location stable
8. **Insights + Limitations** — what worked, what didn't, future work