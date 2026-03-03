# SafeNet Data Pipeline

Replicates the SafeNet paper's feature engineering: raw earthquake catalog → 282-feature pickle for the model.

## Quick Start

```bash
python3 split_data.py    # Split raw data into training/testing CSVs
python3 pipeline.py      # Generate training_output.pickle & testing_output.pickle
python3 validate.py      # Compare testing_output against reference pickle
```

## Pipeline Overview

```
data/1970-2021_11_EARTH_final_with_patchnum.csv
        │
        ├── split_data.py
        │       ├── data/training_data.csv  (1970-2010)
        │       └── data/testing_data.csv   (2002-2021)
        │
        └── pipeline.py
                ├── data/training_output.pickle  (target years 1970-2010)
                └── data/testing_output.pickle   (target years 2011-2020)
```

## Feature Engineering (282 columns)

Each feature vector describes seismic activity within a **12-month window** ending Nov 16 of year Y (Nov 17 Y-1 → Nov 16 Y) following [SafeNet's catalog indicators](https://www.nature.com/articles/s41598-025-93877-7/tables/1)

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

Each pickle contains:
- `eq_data`: list of arrays, one per target year, each shaped `(10, 86, 282)`
  - **10** = history years (Y-9 to Y)
  - **86** = patch 0 (general map) + patches 1-85 (individual regions from `png_list_to_patchxy.csv`)
  - **282** = normalized features (float32, min-max scaled to [0,1])
- `png`: placeholder (empty list)

## Normalization

Per-column min-max normalization computed globally across **all regions and all years** in the dataset:

```
normalized = (value - col_min) / (col_max - col_min)
```

Parameters are saved to `*_norm_params.json` alongside each pickle.
