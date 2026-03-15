"""
CEED Dataset Handler

Loads the CEED dataset from HuggingFace:
https://huggingface.co/datasets/AI4EPS/CEED

And handles:
- metadata catalog building
- catalog access (year / locations)
- lazy waveform pointer index
- PyTorch Dataset wrapper

Designed for metadata-first exploration and ML pipelines.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import h5py
from huggingface_hub import hf_hub_download
import shutil

class CEEDdataset:
    """
    CEED dataset handler for metadata catalog and lazy waveform loading.
    """

    def __init__(self, dataset_name="CEED", base_path=None, waveform_base_path=None, 
                 catalog_path=None):
        """
        Args:
            dataset_name (str, default="CEED"):
                Name of dataset (for generalization, currently only supports "CEED")
            base_path (Path):
                Where to save/load metadata parquet
            waveform_base_path (Path):
                Base path where HDF5 waveform files are stored
        """

        self.dataset_name = dataset_name
        self.base_path = Path(base_path) if base_path else Path(f"data/{dataset_name}")
        self.catalog_path = Path(catalog_path) if catalog_path else Path(f"{self.base_path}/catalog.parquet")
        self.waveform_base_path = Path(waveform_base_path) if waveform_base_path else Path(f"data/{dataset_name}")
        self.catalog = None
        self.pointer_index = None

    # -----------------------------
    # CATALOG BUILDING
    # -----------------------------
    @staticmethod
    def extract_year(event_time):
        return datetime.fromisoformat(str(event_time)).year


    """
    OLD METHOD - Dataset scripts are no longer supported by HuggingFace

    def build_catalog(self, max_events=None, streaming=True):
        print("Building metadata catalog...")
        dataset = load_dataset(
            "AI4EPS/CEED",
            name="event",
            split="train",
            streaming=streaming # stream only, `trust_remote_code` is not supported anymore.
        )

        rows = []
        for i, event in enumerate(tqdm(dataset)):
            lat, lon, depth = event["event_location"]
            rows.append({
                "event_time": event["event_time"],
                "year": self.extract_year(event["event_time"]),
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth,
                "n_stations": len(event["station_location"]),
                "n_phases": len(event["phase_time"]),
            })
            if max_events and i >= max_events:
                break

        df = pd.DataFrame(rows)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.catalog_path)
        self.catalog = df
        print(f"Catalog saved to {self.catalog_path}, events: {len(df)}")
    """

    def download_metadata_csv(self)-> str:
        """
        Download events.csv directly from HF Hub.
        
        Returns:
            local_csv (str): Path to the downloaded CSV file
        """
        self.base_path.mkdir(exist_ok=True, parents=True)
        local_csv = hf_hub_download(
            repo_id="AI4EPS/CEED",
            filename="events.csv",
            repo_type="dataset",
            local_dir=self.base_path,
            local_dir_use_symlinks=False
        )
        
        print(f"csv is: \n {local_csv}")
        print(f"Downloaded events.csv to {local_csv}")
        return local_csv

    def build_catalog(self):
        """
        Build Parquet catalog from downloaded CSV.
        """
        csv_path = self.base_path / "events.csv"
        if not csv_path.exists():
            csv_path = self.download_metadata_csv()

        df = pd.read_csv(csv_path)
        df["event_time"] = pd.to_datetime(df["event_time"])
        df["year"] = df["event_time"].dt.year
        df.to_parquet(self.catalog_path)
        self.catalog = df
        print(f"Catalog saved to {self.catalog_path}, events: {len(df)}")

    
    # -----------------------------
    # CATALOG ACCESS
    # -----------------------------
    def load_catalog(self)-> pd.DataFrame:
        """Load catalog from Parquet file if not already loaded."""
        
        if self.catalog_path is None: # parquet not created
            self.build_catalog()
            
        elif self.catalog is None: # parquet exists but not loaded
            self.catalog = pd.read_parquet(self.catalog_path)
        return self.catalog

    def get_years(self)-> list:
        """Get all available years from the catalog."""
        if self.catalog is None:
            self.load_catalog()
        return sorted(self.catalog.year.unique())

    def get_year(self, year:int)-> pd.DataFrame:
        """Get events for a specific year."""
        if self.catalog is None:
            self.load_catalog()
        return self.catalog[self.catalog.year == year]

    def get_time_range(self, start:int, end:int)-> pd.DataFrame:
        """Get events within a specific time range (inclusive)."""
        if self.catalog is None:
            self.load_catalog()
        return self.catalog[(self.catalog.year >= start) & (self.catalog.year <= end)]

    def get_locations(self)-> pd.DataFrame:
        """Get latitude, longitude, and depth_km for all events in the catalog."""
        if self.catalog is None:
            self.load_catalog()
        return self.catalog[["latitude", "longitude", "depth_km"]]

    # -----------------------------
    # POINTER INDEX
    # -----------------------------
    def build_pointer_index(self) -> dict:
        """
        Map event_id -> waveform path (lazy loading)
        
        Returns:
            pointers (dict):
                dictionnary of form {event_id: waveform_path}
        """
        if self.catalog is None:
            self.load_catalog()
        pointers = {}
        for i, row in self.catalog.iterrows():
            year = row["year"]
            event_id = f"{year}_{i}"
            
            # TODO: This is an example path for the waveform files, adjust if needed.
            pointers[event_id] = str(self.waveform_base_path / f"{year}/event_{i}.h5")
            
        self.pointer_index = pointers
        return pointers

    # -----------------------------
    # PYTORCH DATASET WRAPPER
    # -----------------------------
    class CEEDTorchDataset(Dataset):
        """
        PyTorch Dataset wrapper for lazy waveform loading
        
        Each item returns:
        - waveform (Tensor): Seismic waveform data
        - magnitude (Tensor): Event magnitude (or 0.0 if not available)
        - event_loc (tuple): (latitude, longitude, depth_km) from HDF5
        """

        def __init__(self, pointer_index, event_ids):
            self.pointer_index = pointer_index
            self.event_ids = event_ids

        def __len__(self)-> int:
            return len(self.event_ids)

        def __getitem__(self, idx:int)-> tuple[torch.tensor, torch.tensor, tuple]:
            """
            Get waveform and metadata for a given event index.
            
            - waveform (torch.tensor): Seismic waveform data
            - magnitude (torch.tensor): Event magnitude (or 0.0 if not available)
            - event_loc (tuple): (latitude, longitude, depth_km) from HDF
            
            Args:
                idx (int): Index of the event in the dataset
                
            Returns:
                item (tuple): (waveform, magnitude, event_loc)
            """
            
            event_id = self.event_ids[idx]
            h5_path = self.pointer_index[event_id]
            with h5py.File(h5_path, "r") as f:
                waveform = f["waveform"][:]
                magnitude = f.attrs.get("magnitude", 0.0)
                event_loc = f.attrs.get("event_location", (0.0, 0.0, 0.0))
            waveform = torch.tensor(waveform, dtype=torch.float32)
            magnitude = torch.tensor(magnitude, dtype=torch.float32)
            return waveform, magnitude, event_loc


    # ----------------------------
    # Fault map helpers
    # ----------------------------
    def get_events_by_year(self, year)-> pd.DataFrame:
        """
        Get earthquake events for a specific year from the catalog.
        
        Args:
            year (int): Year to filter events by
        Returns:
            df_year (pd.DataFrame): 
                Df of events for the specified year
        """
        df = self.load_catalog()
        return df[df.year == year]
    
      
    
# ---------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------
def main() -> dict:
    # Initialize
    # ceed = CEEDdataset()
    
    # Build catalog (downloads CSV and saves Parquet)
    # ceed.build_catalog()
    
    # Access events from 2010
    # events_2010 = ceed.get_events_by_year(2010)
    
    # List available years
    # years = ceed.get_years()

    # Build waveform pointer index
    # pointers = ceed.build_pointer_index()

    # Create PyTorch dataset for training
    # train_ids = list(events_2010.index)
    # torch_dataset = ceed.CEEDTorchDataset(ceed.pointer_index, train_ids)

    return ceed

if __name__ == "__main__":
    main()