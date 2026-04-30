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
(5, 400, 400)
```

Channels:

```
0 sedimentary lithology
1 volcanic lithology
2 crystalline lithology
3 distance to nearest fault
4 earthquake magnitude density
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
(N_patches, 5, 50, 50)
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
400 × 400
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

```python
uv run ceed_loader.py #if you have not downloaded the dataset
uv run preprocess_full_pipeline.py
```

This will:
1. Run ceed_maps_pipeline to generate png_data, which will:
- download metadata if needed
- build catalog.parquet
- download shapefiles if needed
- generate yearly tensors
- extract training patches
2. Run Safe-net style processing to generate eq_data
3. Combine png_data and eq_data to form pickle

Outputs are written to:

```
data/CEED/processed
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


# Design Notes

* Metadata is stored in **Parquet** for fast filtering.
* Spatial layers use **identical raster transforms** to guarantee alignment.
* Waveform data is **not loaded locally**; pointer indices allow lazy loading during training.
* Tensors are saved as `.npy` for fast loading in ML pipelines.
