"""
Download USGS fault and state geology shapefiles for California.

Could add more shapefiles in the future, e.g. seismicity, topography, diff countries, etc.

US Faults:
https://www.usgs.gov/natural-hazards/earthquake-hazards/faults

US Geology (only California is downloaded):
https://mrdata.usgs.gov/geology/state/
"""

import os
import requests
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'california' 
FAULTS_DIR = DATA_DIR / "faults"
GEOLOGY_DIR = DATA_DIR / "geology"

FAULTS_DIR.mkdir(parents=True, exist_ok=True)
GEOLOGY_DIR.mkdir(parents=True, exist_ok=True)


def download_and_unzip(url:str, extract_to:str):
    """
    Download a zip file and extract it to a directory.
    Args:
        url (str): URL of the zip file to download
        extract_to (Path): Directory to extract the contents to
    """
    
    local_zip = extract_to / "tmp.zip"
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {local_zip}, extracting ...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    local_zip.unlink()
    print(f"Extraction complete: {extract_to}")


def download_faults():
    """
    Downloads the USGS Quaternary Fault and Fold Database (shapefile)
    Only California is needed for CEED, but download the full Qfaults dataset
    """
    # USGS Qfaults (2022 update) - Full database
    url = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip"
    download_and_unzip(url, FAULTS_DIR)
    print(f"Fault shapefile saved at {FAULTS_DIR}")


def download_geology():
    """
    Downloads USGS State Geology shapefile for California
    """
    # MRData CA Geology shapefile (2021 update)
    url = (
        "https://mrdata.usgs.gov/geology/state/shp/CA.zip"
    )
    download_and_unzip(url, GEOLOGY_DIR)
    print(f"Geology shapefile saved at {GEOLOGY_DIR}")


if __name__ == "__main__":
    download_faults()
    download_geology()
    print("All shapefiles downloaded successfully.")