import pandas as pd
import glob
import os

RAW_DATA_DIR = os.path.expanduser("/home/du2/22CS30034/karan/GIS/lubesh")
rest_file = glob.glob(os.path.join(RAW_DATA_DIR, "*RestPoints*.tab"))[0]

# Load the dataset
df_rest = pd.read_csv(rest_file, sep="\t", low_memory=False)

# Count unique buildings and get their names
num_buildings = df_rest["Building_Name"].nunique()
building_names = df_rest["Building_Name"].dropna().unique()

print(f"Total unique buildings: {num_buildings}")
print("\nBuilding Names:")
for name in sorted(building_names):
    print(f"- {name}")