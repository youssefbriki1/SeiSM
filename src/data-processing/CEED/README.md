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