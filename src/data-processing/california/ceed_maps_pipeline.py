"""
Runs the full CEED spatial dataset pipeline:
- Downloads shapefiles if missing
- Builds yearly 5-channel tensors
- Splits into SafeNet-style patches
"""

from pathlib import Path
import numpy as np
import subprocess
import sys
from ceed_loader import CEEDdataset
from ceed_maps_builder import CEEDmaps

def main():

    OUTPUT = Path("data/processed/cal_maps")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Ensure shapefiles exist
    # ------------------------------
    FAULTS_FILE = Path("data/faults/SHP/Qfaults_US_Database.shp")
    GEOLOGY_FILE = Path("data/geology/CA_geol_poly.shp")

    if not FAULTS_FILE.exists() or not GEOLOGY_FILE.exists():
        print("Shapefiles not found. Running download_shapefiles.py ...")
        subprocess.run([sys.executable, "download_shapefiles.py"], check=True)

    # ------------------------------
    # Load CEED dataset
    # ------------------------------
    
    # df = CEEDdataset()                                        # from scratch
    df = CEEDdataset(catalog_path="data/CEED/catalog.parquet") # if parquet already built
    
    df.load_catalog()
    # ------------------------------
    # Initialize map builder
    # ------------------------------
    builder = CEEDmaps(
        df,
        faults_path=FAULTS_FILE,
        geology_path=GEOLOGY_FILE
    )

    # ------------------------------
    # Loop over years
    # ------------------------------
    for year in df.get_years():

        print(f"Processing year {year} ...")
        tensor = builder.build_year_tensor(year)
        patches = builder.extract_patches(tensor)

        np.save(OUTPUT / f"tensor_{year}.npy", tensor)
        np.save(OUTPUT / f"patches_{year}.npy", patches)

        print(
            f"{year}: tensor {tensor.shape} | patches {patches.shape}"
        )

if __name__ == "__main__":
    main()