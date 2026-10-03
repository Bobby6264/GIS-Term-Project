import pandas as pd
import os

# ---- find the csv.gz ----
csv_path = None
for root, _, files in os.walk("data/kth"):
    for f in files:
        if f.endswith(".csv.gz"):
            csv_path = os.path.join(root, f)
            break
    if csv_path:
        break

if not csv_path:
    print("❌ No .csv.gz found under data/kth/")
    raise SystemExit

print(f"📄 CSV: {csv_path}")
print(f"   size: {os.path.getsize(csv_path)/(1024*1024):.1f} MB")

# ---- inspect first rows ----
df = pd.read_csv(csv_path, nrows=5)
print(f"\nColumns ({len(df.columns)}): {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.to_string())

# ---- count rows (compressed-safe, streams through) ----
# quick count using chunksize (avoids loading 400MB into memory)
n = 0
for chunk in pd.read_csv(csv_path, chunksize=500_000):
    n += len(chunk)
print(f"\nTotal rows: {n:,}")

# ---- find + show APlocations.txt ----
ap_path = None
for root, _, files in os.walk("data/kth"):
    for f in files:
        if f.lower().startswith("aplocations"):
            ap_path = os.path.join(root, f)
            break
    if ap_path:
        break

if ap_path:
    print(f"\n📄 {ap_path} (first 10 lines):")
    with open(ap_path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            print(f"  {line.rstrip()}")
            if i >= 9:
                break
else:
    print("\n⚠️ APlocations.txt not found under data/kth/")