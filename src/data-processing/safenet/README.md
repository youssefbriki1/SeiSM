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
        ├─ data/training_output.pickle  (features: target years 1979–2010)
        ├─ data/training_labels.pickle  (labels:   1980–2011)
        ├─ data/testing_output.pickle   (features: target years 2011–2020)
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

## Output Structure

Each feature pickle contains:

- **`eq_data`**: list of arrays, one per target year, each `(10, 86, 282)`
  - `10` — history years (Y-9 to Y)
  - `86` — patch 0 (general map) + patches 1–85 (regions via `png_list_to_patchxy.csv`)
  - `282` — min-max normalized features (float32, [0,1])
- **`png`**: placeholder (empty list)

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
# CEED Dataset Processing

This folder contains tools for **metadata extraction, fault maps, and ML dataset preparation** for the CEED dataset from Hugging Face.

Dataset URL: https://huggingface.co/datasets/AI4EPS/CEED


## Overview

The CEED dataset contains earthquake events, picks, and waveform data. However, the Hugging Face dataset loader (`CEED.py`) is no longer supported by the `datasets` library because executing remote dataset scripts has been deprecated.

To avoid this issue and to keep the workflow lightweight, we directly download the **metadata CSV (`events.csv`)** from the Hugging Face Hub and build a local catalog from it.

This approach allows:

- metadata analysis without downloading waveform data
- generation of earthquake fault maps
- fast filtering and splitting by year
- preparation of datasets for machine learning
- compatibility with HPC training workflows later

Only about **~90MB of metadata** needs to be downloaded initially.

Waveforms can be loaded later during HPC training.


## Directory Structure

Example expected structure:

```
src/
  data/
    CEED/
      events.csv
      ceed_catalog.parquet
  data-processing/
    ceed_dataset.py
```


## Workflow

### 1. Download Metadata

```python
from ceed_dataset import CEEDDataset

ceed = CEEDDataset()
ceed.download_metadata_csv()
```

This downloads `events.csv` directly from the Hugging Face Hub.
The file contains metadata for all earthquake events and is roughly 90MB.


### 2. Build Metadata Catalog

```python
ceed.build_catalog()
```

This converts the CSV metadata into a **Parquet catalog**:

```
data/CEED/ceed_catalog.parquet
```

Parquet is used because it allows:

- fast filtering
- efficient storage
- easy integration with pandas and ML pipelines

Typical columns include:

- `event_time`
- `year`
- `latitude`
- `longitude`
- `depth_km`
- `magnitude`
- other event metadata


### 3. Query Metadata

Example: get events from a specific year.

```python
events_2010 = ceed.get_events_by_year(2010)
```

Get all available years:

```python
years = ceed.get_years()
```


### 4. Fault Map Generation

Fault maps can be created using **GMT v6**.

Each earthquake event is plotted as a circle:

- position = latitude / longitude
- size = magnitude

Example workflow:

1. extract events from catalog
2. export coordinates and magnitude
3. generate GMT map

Example extraction:

```python
events = ceed.get_events_by_year(2010)

coords = events[["latitude", "longitude", "magnitude"]]
coords.to_csv("events_2010.csv", index=False)
```

GMT can then plot circles using magnitude as radius.


### 5. Pointer Index for Machine Learning

Waveform data is extremely large and should not be loaded locally.

Instead, we build a **pointer index** that maps event IDs to waveform file locations.

```python
pointers = ceed.build_pointer_index()
```

This produces a structure like:

```
{
  "2010_0001": "data/CEED/2010/event_0001.h5",
  "2010_0002": "data/CEED/2010/event_0002.h5",
  ...
}
```

Later, HPC training scripts can load waveforms lazily.


### 6. PyTorch Dataset

A PyTorch dataset wrapper allows waveforms to be loaded on demand.

Example:

```python
train_ids = list(pointers.keys())[:1000]

dataset = ceed.TorchDataset(pointers, train_ids)
```

This dataset returns:

```
waveform tensor
magnitude
event location
```

It can be used with a standard PyTorch `DataLoader`.



## Future Generalization

This project will likely use multiple seismic datasets.

To support this, the current implementation separates:

- dataset-specific loader (`CEEDDataset`)
- generic metadata catalog format
- generic ML dataset wrapper
- generic fault map generation

Future datasets can reuse the same architecture.

Possible future structure:

```
datasets/
  ceed_dataset.py
  other_dataset.py
catalogs/
  catalog_builder.py
faultmaps/
  map_generator.py
```

---

## Sources

CEED dataset repository  
https://huggingface.co/datasets/AI4EPS/CEED

SeisBench documentation  
https://seisbench.readthedocs.io

Hugging Face Hub API  
https://huggingface.co/docs/huggingface_hub

Parquet format documentation  
https://parquet.apache.org