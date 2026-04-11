"""
Builds spatial tensors from CEED earthquake catalog and USGS geology and fault data.

Output:
    yearly tensors (5,512,512)
    SafeNet-style patches
"""
from pathlib import Path

from ceed_loader import CEEDdataset
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from eq_map_generation import ImageProcessing


class CEEDmaps:
    """
    CEED and USGS data handler for building spatial tensors in the SafeNet format.
    
    This is 5-channel: 1 for earthquake density, 1 for fault distance, 3 for lithology classes.
    Then split into patches for ML pipelines.
    """
    def __init__(self, ceed_dataset:CEEDdataset, faults_path:str, geology_path:str, 
                 bbox:tuple=(-125, 32, -113, 42), grid_size:int=400, patch_size:int=50,
                 stride:int=50,):
        """
        Args:
            ceed_dataset (CEEDdataset):
                Instance of CEEDdataset for metadata access
            faults_path (str):
                Path to USGS fault shapefile
            geology_path (str):
                Path to USGS geology shapefile
            bbox (tuple, default=(-125, 32, -113, 42)):
                Bounding box for the region of interest (xmin, ymin, xmax, ymax)
            grid_size (int, default=400):
                Size of the output spatial tensor (grid_size x grid_size)
            patch_size (int, default=50):
                Size of the extracted patches (patch_size x patch_size)
            stride (int, default=50):
                Stride for patch extraction (default is non-overlapping)
        """

        self.ds = ceed_dataset

        self.faults_path = faults_path
        self.geology_path = geology_path

        self.xmin, self.ymin, self.xmax, self.ymax = bbox

        self.grid_size = grid_size
        self.patch_size = patch_size
        self.stride = stride

        self.dx = (self.xmax - self.xmin) / grid_size
        self.dy = (self.ymax - self.ymin) / grid_size

        self._load_static_layers()

    # -----------------------------
    # coordinate helpers
    # -----------------------------

    def lon_to_x(self, lon: float)-> int:
        return ((lon - self.xmin) / self.dx).astype(int)

    def lat_to_y(self, lat: float)-> int:
        return ((self.ymax - lat) / self.dy).astype(int)

    # -----------------------------
    # static layers
    # -----------------------------

    def _load_static_layers(self):
        """Load static layers (fault distance and lithology) once during initialization."""

        self.fault_distance = self._build_fault_distance()
        self.lithology = self._build_lithology()

    # -----------------------------
    # FAULT DISTANCE
    # -----------------------------

    def _build_fault_distance(self)-> np.ndarray:
        """
        Build a distance map to the nearest fault line using USGS fault shapefile.
         - Reads fault geometries, rasterizes them, and computes distance transform.
         - Normalizes distance to [0,1] range for ML input.
         
        Returns:
            2D numpy array of shape (grid_size, grid_size) with normalized distances to faults.
        """
        # Skip loading DBF attributes to avoid date parsing crashes on SLURM
        # Datasets are natively EPSG:4326. Avoiding .to_crs() bypasses pyproj entirely.
        faults = gpd.read_file(self.faults_path, columns=[])

        bbox = box(self.xmin, self.ymin, self.xmax, self.ymax)
        faults = faults.clip(bbox)

        transform = rasterio.transform.from_bounds(
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
            self.grid_size,
            self.grid_size,
        )

        shapes = ((g, 1) for g in faults.geometry)

        fault_mask = rasterize(
            shapes,
            out_shape=(self.grid_size, self.grid_size),
            transform=transform,
        )

        fault_mask = 1 - fault_mask

        return fault_mask

        # distance = distance_transform_edt(1 - fault_mask)

        # distance /= distance.max()

        # return distance

    # -----------------------------
    # GEOLOGY → 3 CHANNELS
    # -----------------------------

    def _classify_lithology(self, row) -> int:
        """Classify lithology into 3 classes (dimensions)"""

        text = str(row).lower()

        if "sediment" in text:
            return 0

        if "volcan" in text:
            return 1

        return 2

    def _build_lithology(self)-> np.ndarray:
        """
        Build a lithology map using USGS geology shapefile.
        - Reads geology polygons, classifies them into 3 classes, and rasterizes each class into separate channels.
        
        Returns:
            3D numpy array of shape (3, grid_size, grid_size) with binary masks for each lithology class.    
        """
        # Skip non-lithology attributes to avoid DBF date parsing crashes on SLURM. 
        # The file is natively EPSG:4326. 
        gdf = gpd.read_file(self.geology_path, columns=['ORIG_LABEL', 'GENERALIZE', 'SGMC_LABEL'])

        bbox = box(self.xmin, self.ymin, self.xmax, self.ymax)
        gdf = gdf.clip(bbox)

        transform = rasterio.transform.from_bounds(
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
            self.grid_size,
            self.grid_size,
        )

        classes = gdf.apply(self._classify_lithology, axis=1)

        layers = []

        for c in range(3):

            shapes = (
                (geom, 1)
                for geom, cl in zip(gdf.geometry, classes)
                if cl == c
            )

            raster = rasterize(
                shapes,
                out_shape=(self.grid_size, self.grid_size),
                transform=transform,
            )

            layers.append(raster)

        return np.stack(layers)

    # -----------------------------
    # EARTHQUAKE GAUSSIAN MAP
    # -----------------------------

    # Old way of generating earthquake distribution map, switched to ImageProcessing to align with safenet-style
    def build_earthquake_map(self, events: pd.DataFrame)-> np.ndarray:
        """
        Build a Gaussian map of earthquake locations.
        
        Args:
            events (DataFrame):
                DataFrame containing earthquake events with 'latitude', 'longitude', and 'magnitude' columns
        
        """
        

        layer = np.zeros((self.grid_size, self.grid_size))

        x = self.lon_to_x(events.longitude.values)
        y = self.lat_to_y(events.latitude.values)

        mags = events.magnitude.values

        for xi, yi, m in zip(x, y, mags):

            if 0 <= xi < self.grid_size and 0 <= yi < self.grid_size:

                layer[yi, xi] += m

        # Gaussian smoothing (SafeNet-like kernel)
        layer = gaussian_filter(layer, sigma=1.5)

        return layer

    # -----------------------------
    # YEAR TENSOR
    # -----------------------------

    def build_year_tensor(self, year)-> np.ndarray:
        """Build a 5-channel spatial tensor for a specific year"""

        CEED_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'california' 
        # events = self.ds.get_events_by_year(year)
        event_csv_path = CEED_DIR / 'events_preprocessed_1987_2010.csv'
        if(year > 2010):
            event_csv_path = CEED_DIR / 'events_preprocessed_2002_2020.csv'
        
        EQ_map = ImageProcessing(
            map_path="map_outline.jpg",
            event_csv_path=event_csv_path,
            patch_csv_path= CEED_DIR / 'png_list_to_patchxy.csv',
            cols= int(self.grid_size/self.patch_size),
            rows = int(self.grid_size/self.patch_size),
            width= self.grid_size,
            height = self.grid_size,
            patch_size= self.patch_size,
            padding = 0
        )

        # eq_layer = self.build_earthquake_map(events)
        eq_layer = EQ_map.generate_eq_map(year)

        tensor = np.vstack(
            [
                self.lithology,
                self.fault_distance[np.newaxis, :, :],
                eq_layer[np.newaxis, :, :],
            ]
        )

        return tensor

    # -----------------------------
    # PATCH EXTRACTION
    # -----------------------------

    def extract_patches(self, tensor):

        C, H, W = tensor.shape

        patches = []

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):

                patch = tensor[
                    :,
                    y : y + self.patch_size,
                    x : x + self.patch_size,
                ]

                patches.append(patch)

        return np.array(patches)
    
