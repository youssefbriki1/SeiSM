# SafeNet Data Pipeline

Replicates the [SafeNet paper](https://www.nature.com/articles/s41598-025-93877-7)'s feature engineering: raw earthquake catalog → 282-feature pickle + labeled ground truth.

## Quick Start

```bash
python3 split_data.py                # Split raw data into training/testing CSVs
python3 pipeline.py                  # Generate feature pickles + label pickles
python3 validate.py                  # Validate features against reference
python3 validate_labeled_data.py     # Validate labels against reference (100% match ✓)
```

## Pipeline Overview

```
data/1970-2021_11_EARTH_final_with_patchnum.csv
    │
    ├─ split_data.py
    │   ├─ data/training_data.csv       (1970–2010)
    │   └─ data/testing_data.csv        (2002–2021)
    │
    └─ pipeline.py
        ├─ data/training_output.pickle  (features: target years 1979–2010 & maps from 1971-2011)
        ├─ data/training_labels.pickle  (labels:   1980–2011)
        ├─ data/testing_output.pickle   (features: target years 2011–2020  & maps from 2002-2021)
        └─ data/testing_labels.pickle   (labels:   2012–2021)
```

## Feature Engineering (282 columns)

Each feature vector covers a **12-month sliding window** (Nov 17 Y-1 → Nov 16 Y), following [SafeNet's catalog indicators](https://www.nature.com/articles/s41598-025-93877-7/tables/1).

| Columns | Count | Description |
|---------|-------|-------------|
| 0–64 | 65 | Top 5 magnitudes: yearly (5) + per month (5×12) |
| 65–129 | 65 | Lunar dates of top 5 magnitudes, normalized via `sin(x/N·2π)+2` |
| 130–177 | 48 | Monthly event counts in 4 mag bins: [0,3], (3,5], (5,7], >7 |
| 178–201 | 24 | Monthly depth counts: >70km and ≤70km |
| 202–215 | 14 | b-value (G-R relation): yearly + 12 months + latest 100 events |
| 216–229 | 14 | a-value (G-R relation): yearly + 12 months + latest 100 events |
| 230–243 | 14 | Mean magnitude: yearly + 12 months + latest 100 events |
| 244–256 | 13 | Std deviation of magnitude: yearly + 12 months |
| 257–269 | 13 | P(M≥6): yearly + 12 months |
| 270 | 1 | a/b ratio for latest 100 events |
| 271 | 1 | MSE of G-R model for latest 100 events |
| 272 | 1 | Time diff between current event and 100th prior event |
| 273 | 1 | RMS earthquake energy (E=10^(1.5M+4.8)) of latest 100 |
| 274–277 | 4 | Mean interval time per mag range for latest 100 |
| 278–281 | 4 | SD/mean interval ratio per mag range for latest 100 |

## Map Processing (5 channels)
Channel 0 - 2: R, G, B representation of the original map (static)
Channel 3: China fault map (static)
Channel 4: Earthquake distribution map (changes year-by-year)

Channel 0-3 are loaded directly from Safent's provided dataset
Channel 4 are dynamically calculated in image_process.py

You can visualize different channels of the map with map_visualization.py

## Output Structure

Each feature pickle contains:

- **`eq_data`**: list of arrays, one per target year, each `(10, 86, 282)`
  - `10` — history years (Y-9 to Y)
  - `86` — patch 0 (general map) + patches 1–85 (regions via `png_list_to_patchxy.csv`)
  - `282` — min-max normalized features (float32, [0,1])
- **`png`**: list of arrays, one per calendar year from `norm_start` to `target_years[-1]+1`, each `(85, 50, 50, 5)`
  - `85` — regions (no global token for images)
  - `50×50` — spatial resolution per patch
  - `5` — channels: R/G/B geology map (0-2), fault map grayscale (3), earthquake distribution (4)

Normalization is per-column min-max across all regions and years:
```
normalized = (value - col_min) / (col_max - col_min)
```
Parameters saved to `*_norm_params.json`.

## Labels (Ground Truth)

Features at year X predict the earthquake magnitude class at year **X+1**.

| Class | Magnitude Range |
|-------|----------------|
| 0 | 0 ≤ M < 5 |
| 1 | 5 ≤ M < 6 |
| 2 | 6 ≤ M < 7 |
| 3 | M ≥ 7 |

Each label pickle contains a list of `(85,)` int arrays — one class per patch per year.

| Dataset | Feature Years | Label Years |
|---------|--------------|-------------|
| Training | 1979–2010 | 1980–2011 |
| Testing | 2011–2020 | 2012–2021 |



---
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
