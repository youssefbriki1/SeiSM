# CEED Spatial Tensor Pipeline

Tools for converting the **CEED earthquake catalog** into **spatial tensors and patches** for deep learning models.

Dataset: https://huggingface.co/datasets/AI4EPS/CEED

The pipeline builds **California earthquake feature maps** from CEED metadata and USGS geospatial datasets. These maps can be used as input to CNN models (e.g., ResNet) for earthquake forecasting or spatial seismicity modeling.


# What the Pipeline Produces

For each CEED year (1987–2020):

```
tensor_YEAR.npy
patches_YEAR.npy
```

Example:

```
tensor_2001.npy
patches_2001.npy
```

### Yearly Tensor

Shape:

```
(5, 512, 512)
```

Channels:

```
0 earthquake magnitude density
1 distance to nearest fault
2 sedimentary lithology
3 volcanic lithology
4 crystalline lithology
```

### Patches

Each yearly tensor is split into training patches.

Example configuration:

```
patch_size = 64
stride = 64
```

Output shape:

```
(N_patches, 5, 64, 64)
```


## Data Sources

CEED earthquake catalog:

```
events.csv (~90MB metadata)
```

USGS spatial datasets:

Fault database
https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip

California geology
https://mrdata.usgs.gov/geology/state/shp/CA.zip

The pipeline downloads shapefiles automatically if missing.


## Spatial Grid

All spatial layers are rasterized onto the same grid.

Bounding box:

```
lon: -125 → -113
lat:  32 → 42
```

Resolution:

```
512 × 512
```


## Directory Structure

```
data/

  CEED/
    events.csv
    catalog.parquet

  faults/
    Qfaults_US_Database.shp

  geology/
    state_geology.shp

  processed/
    ceed_maps/
      tensor_YYYY.npy
      patches_YYYY.npy
```

## Pipeline Overview

The pipeline performs four steps:

```
CEED metadata → spatial features → yearly tensors → training patches
```

1. Download CEED metadata (`events.csv`)
2. Build Parquet catalog (`catalog.parquet`)
3. Generate yearly spatial tensors
4. Extract patches

The pipeline is **idempotent**:

```
if CSV missing → download
if Parquet missing → build
otherwise → load existing data
```

## Running the Pipeline

Run the full pipeline:

```
python ceed_maps_pipeline.py
```

This will:


- download metadata if needed
- build catalog.parquet
- download shapefiles if needed
- generate yearly tensors
- extract training patches

Outputs are written to:

```
data/processed/ceed_maps/
```

## Using the Dataset Class

```python
from ceed_dataset import CEEDdataset

ceed = CEEDdataset(catalog_path="data/CEED/catalog.parquet")
ceed.load_catalog()
```

Example queries:

```python
events_2010 = ceed.get_events_by_year(2010)
years = ceed.get_years()
```


## Visualization

A notebook is provided for inspecting tensors and patches:

```
visualize_ceed_tensors.ipynb
```

The notebook allows visualization of:

```
5-channel yearly tensors
patch grids
individual channels
```

Example tensor visualization:

```
(5, 512, 512)
```

Example patch visualization:

```
(5, 64, 64)
```


# Design Notes

* Metadata is stored in **Parquet** for fast filtering.
* Spatial layers use **identical raster transforms** to guarantee alignment.
* Waveform data is **not loaded locally**; pointer indices allow lazy loading during training.
* Tensors are saved as `.npy` for fast loading in ML pipelines.

---

# Possible Extensions

Future spatial channels may include:

```
strain rate
topography
fault slip rate
seismic moment release
10-year sliding seismicity windows
```

These features can be added as additional tensor channels.
