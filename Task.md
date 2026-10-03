# Spatio-Temporal Hotspot Analysis of Campus Events
## Using DBSCAN, ST-DBSCAN and KDE

---

## 1. PROJECT REQUIREMENT

The course project requires us to perform a **spatio-temporal hotspot
analysis** of campus events using **density-based spatial clustering
(DBSCAN)** and **Kernel Density Estimation (KDE)**.

The goal is to:
- Identify **where** events concentrate (spatial hotspots)
- Identify **when** they concentrate (temporal patterns)
- Combine both into a **spatio-temporal** view

No KGP-specific dataset was provided. The TA confirmed that any publicly
available dataset with spatial + temporal attributes may be used.

---

## 2. WHY WE GENERATED SYNTHETIC DATA

We had no access to real campus event logs. Public campus event datasets
(such as UNIST Wi-Fi traces) are either license-gated, heavily
anonymized, or require significant preprocessing not feasible in the
available time.

**We therefore built a synthetic campus event generator** with:
- 3 known hotspot locations (Library, Cafeteria, Sports Complex)
- Distinct temporal patterns per hotspot (e.g. Cafeteria peaks at
  lunch and dinner; Sports peaks morning and evening)
- 500 uniform noise events scattered across the campus area

**Why this is valid:**
- Synthetic data with **known ground truth** is standard practice for
  validating clustering algorithms
- It lets us quantitatively verify whether DBSCAN / ST-DBSCAN recover
  the correct clusters and correct temporal splits
- Real data has no ground truth — so we cannot measure correctness on
  it alone

**Generator details:**
- 5,000 total events across 30 simulated days
- Hotspot coordinates chosen ~700–900 m apart to ensure clean spatial
  separation
- Per-hotspot spatial spread: 25–35 m (Gaussian)
- Events assigned hours drawn only from each hotspot's peak-hour list

---

## 3. WHAT WE TESTED (AND WHY)

### 3.1 Synthetic campus data (primary validation)

Because we know the ground truth, we can verify:

| Test | Expected | What we check |
|------|----------|---------------|
| Spatial DBSCAN | Recover 3 hotspots | Cluster count = 3 |
| Noise recovery | Label ~500 events as noise | Noise count ≈ 500 |
| ST-DBSCAN | Split each location by time | Cluster count > 3 |
| KDE time slices | Hotspot intensity shifts by hour | Visual inspection |

### 3.2 Chicago crime data (real-world generalization)

Used to demonstrate that the **same pipeline** works on real data with
the same structure (latitude + longitude + timestamp).

**Why Chicago crime?**
- Public, clean, well-documented
- ~50,000 events in 2023 with exact coordinates and timestamps
- Same data structure as campus events: point events with location + time
- Classic benchmark in hotspot analysis literature

**Limitation acknowledged:**
- Crime is not a "campus event" — but the methodology is domain-agnostic
- Purpose is to show the pipeline generalizes, not to analyze crime itself

---

## 4. ALGORITHMS USED

### 4.1 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

- Groups points that are densely packed together (within radius `eps`,
  with at least `minPts` neighbours)
- Points in low-density regions are labelled **noise** (-1)
- Does not require specifying the number of clusters
- Handles arbitrary cluster shapes
- Chosen over K-means because: no `k` required, noise handling built-in,
  no assumption of spherical clusters

**Parameter selection:**
- `eps` chosen from the **k-distance graph knee**
- `minPts` chosen using the rule `minPts ≥ dim + 1`, then tuned

### 4.2 ST-DBSCAN (Spatio-Temporal DBSCAN)

- Same idea as DBSCAN, but the neighbour condition requires closeness
  in **both space AND time**
- Implementation: combine (x, y, hour) into one feature space, scale
  time so `eps_time` hours ≈ `eps_spatial` metres, then run DBSCAN
- Two events at the same location but hours apart are **not** neighbours
- This is what makes the analysis *spatio-temporal* rather than spatial-only

### 4.3 KDE (Kernel Density Estimation)

- Smooths point data into a continuous density surface (heatmap)
- Every event contributes a small Gaussian bump; overlapping bumps create
  high-density regions (hotspots)
- Provides a **continuous** view of hotspots, complementary to
  DBSCAN's **discrete** clusters
- We also compute KDE per time slice to visualise hotspot evolution
  through the day

---

## 5. WHAT EACH PLOT SHOWS

| Figure | Shows |
|--------|-------|
| `fig_kdistance.png` | Sorted distance to k-th nearest neighbour; the knee indicates a good `eps` |
| `fig_dbscan_clusters.png` | Spatial DBSCAN result — discrete clusters + noise (grey) |
| `fig_stdbscan_clusters.png` | Two panels: spatial view of ST-DBSCAN clusters + temporal histogram showing *when* each cluster occurs |
| `fig_kde_allday.png` | KDE density for all events combined |
| `fig_kde_time_slices.png` | KDE density in 4 time windows (Morning / Afternoon / Evening / Night) — hotspot intensity shifts by time |
| `fig_kde_vs_dbscan.png` | Side-by-side: KDE density surface vs DBSCAN hard clusters (complementary views) |
| `fig_chicago_dbscan.png` | DBSCAN applied to Chicago crime — 3 macro-hotspots |
| `fig_chicago_stdbscan.png` | ST-DBSCAN on Chicago crime |
| `fig_chicago_kde.png` | KDE density surface for Chicago crime |

---

## 6. RESULTS SUMMARY

### 6.1 Synthetic campus data

| Method | Clusters | Noise % | Silhouette |
|--------|----------|---------|------------|
| DBSCAN (spatial) | **3** | 8.8% | **0.901** |
| ST-DBSCAN | **6** | 9.4% | **0.548** |

- **Spatial DBSCAN recovered the 3 planted hotspots exactly** — silhouette 0.90
- **ST-DBSCAN found 6 clusters** — each hotspot split into 2 temporal regimes
  - Cafeteria → lunch + dinner (897 + 906 = ~1800)
  - Library → two daytime regimes (905 + 606 = ~1500)
  - Sports → morning + evening (605 + 609 = ~1200)
- **Noise count** (~440) close to the planted 500 noise events
- **KDE time slices** show different hotspots active at different hours

### 6.2 Chicago crime data

| Method | Clusters | Noise % | Silhouette |
|--------|----------|---------|------------|
| DBSCAN (spatial) | 3 | 0.1% | 0.177 |
| ST-DBSCAN | 5 | 2.2% | −0.064 |

- Same pipeline produces **3 macro-hotspots** — structurally consistent
  with the synthetic result
- Silhouette is much lower because real urban crime density varies
  **continuously** — no clean gaps between clusters
- ST silhouette near zero because crime lacks the strong temporal
  periodicity that campus events have

### 6.3 Cross-dataset insight

> Both datasets yield **3 dominant spatial hotspots**, though with very
> different separation quality (0.90 vs 0.18). The pipeline is robust:
> it extracts stable structure even when real-world data violates
> synthetic assumptions.

---


---

## 8. HOW TO RUN (REPRODUCE FROM SCRATCH)

```bash
# 1. setup
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn scipy matplotlib seaborn folium requests

# 2. generate synthetic data
python scripts/generate_data.py

# 3. parameter sensitivity check
python scripts/tune_eps.py

# 4. spatial DBSCAN
python scripts/dbscan_spatial.py

# 5. spatio-temporal DBSCAN
python scripts/stdbscan.py

# 6. KDE analysis
python scripts/kde_analysis.py

# 7. download + analyze Chicago crime
python scripts/download_chicago.py
python scripts/tune_chicago.py
python scripts/chicago_pipeline.py

# 8. consolidate results
python scripts/summarize.py
