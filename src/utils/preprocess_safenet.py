"""
preprocess_safenet.py
---------------------
Simple SafeNet-style preprocessing pipeline.

What it does
------------
1. Load the raw earthquake catalog CSV
2. Clean the data  (bad depth, low magnitude, out-of-bounds events)
3. Bin every event into a 50 × 50 lat/lon grid (China region)
4. Slide a 365-day window (43-day stride) across the full catalog
5. Compute 5 features per (grid cell, window):
      event_count, mean_magnitude, b_value, mean_depth, time_since_last_event
6. Per-cell min-max normalisation fitted on training windows only
7. Save (T, 50, 50, 5) tensor → data/processed/safenet_dataset.h5

Run
---
    python src/preprocess_safenet.py
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_CSV   = Path("data/raw/1970-2021_11_EARTH_final_with_patchnum.csv")
OUTPUT_H5 = Path("data/processed/safenet_dataset.h5")

BOUNDS = (73.0, 18.0, 135.0, 53.0)   # lon_min, lat_min, lon_max, lat_max
N_LAT, N_LON   = 50, 50
WINDOW_DAYS    = 365
STRIDE_DAYS    = 43
MAG_MIN        = 2.0
MIN_DEPTH_KM   = 0.0
MAX_DEPTH_KM   = 700.0
B_MIN_EVENTS   = 10
TRAIN_FRAC     = 0.88
VAL_FRAC       = 0.06

FEATURE_NAMES = ["event_count", "mean_magnitude", "b_value",
                 "mean_depth", "time_since_last_event"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------

def load_and_clean(path: Path) -> pd.DataFrame:
    log.info("Loading %s …", path.name)
    df = pd.read_csv(path)

    # Normalise column names
    aliases = {
        "time":      ["date_time", "time", "datetime", "origin_time"],
        "latitude":  ["latitude",  "lat",  "Latitude"],
        "longitude": ["longitude", "lon",  "lng", "Longitude"],
        "depth_km":  ["depth",     "depth_km", "dep"],
        "magnitude": ["magnitude", "mag",  "Magnitude", "ML", "Mw"],
    }
    rename = {}
    for canon, opts in aliases.items():
        for o in opts:
            if o in df.columns and canon not in df.columns:
                rename[o] = canon
                break
    df = df.rename(columns=rename)

    # Parse timestamps (strip tz so everything is tz-naive)
    df["time"] = (pd.to_datetime(df["time"], utc=True, errors="coerce")
                  .dt.tz_convert(None))

    # Sentinel depth -1 → NaN
    if "depth_km" in df.columns:
        df["depth_km"] = df["depth_km"].where(df["depth_km"] >= 0.0, other=np.nan)

    n0 = len(df)
    df = df.dropna(subset=["time", "latitude", "longitude", "magnitude"])
    df = df[df["magnitude"] >= MAG_MIN]

    if "depth_km" in df.columns:
        bad = (df["depth_km"].notna() &
               ((df["depth_km"] < MIN_DEPTH_KM) | (df["depth_km"] > MAX_DEPTH_KM)))
        df = df[~bad]

    lon_min, lat_min, lon_max, lat_max = BOUNDS
    df = df[(df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max) &
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)]

    df = df.sort_values("time").reset_index(drop=True)
    log.info("Kept %d / %d events after cleaning", len(df), n0)
    return df


# ---------------------------------------------------------------------------
# 2. Assign grid cells from lat/lon
# ---------------------------------------------------------------------------

def assign_grid_cells(df: pd.DataFrame):
    lon_min, lat_min, lon_max, lat_max = BOUNDS
    lat_edges = np.linspace(lat_min, lat_max, N_LAT + 1)
    lon_edges = np.linspace(lon_min, lon_max, N_LON + 1)

    rows = np.clip(np.searchsorted(lat_edges[1:], df["latitude"].values,  "right"), 0, N_LAT - 1)
    cols = np.clip(np.searchsorted(lon_edges[1:], df["longitude"].values, "right"), 0, N_LON - 1)

    df = df.copy()
    df["row"] = rows
    df["col"] = cols
    log.info("Grid %d×%d — row ∈ [%d,%d], col ∈ [%d,%d]",
             N_LAT, N_LON, rows.min(), rows.max(), cols.min(), cols.max())
    return df, lat_edges, lon_edges


# ---------------------------------------------------------------------------
# 3. Sliding windows
# ---------------------------------------------------------------------------

def make_windows(t_start: pd.Timestamp, t_end: pd.Timestamp) -> list:
    wins, t, i = [], t_start, 0
    while t + timedelta(days=WINDOW_DAYS) <= t_end + timedelta(days=1):
        wins.append((i, t, t + timedelta(days=WINDOW_DAYS)))
        t += timedelta(days=STRIDE_DAYS)
        i += 1
    log.info("Generated %d windows (%d-day / %d-day stride)", len(wins), WINDOW_DAYS, STRIDE_DAYS)
    return wins


# ---------------------------------------------------------------------------
# 4. Feature extraction for one window
# ---------------------------------------------------------------------------

def b_value(mags: np.ndarray) -> float:
    m = mags[mags >= MAG_MIN]
    if len(m) < B_MIN_EVENTS or m.mean() <= MAG_MIN:
        return np.nan
    return float(np.log10(np.e) / (m.mean() - MAG_MIN))


def extract_features(df_win: pd.DataFrame, win_end: pd.Timestamp) -> np.ndarray:
    feat = np.full((N_LAT, N_LON, 5), np.nan, dtype=np.float32)
    feat[:, :, 0] = 0.0                     # event_count = 0 by default
    feat[:, :, 4] = float(WINDOW_DAYS)      # no events → full window length

    if df_win.empty:
        return feat

    grp = df_win.groupby(["row", "col"], sort=False)

    agg = grp.agg(count=("magnitude", "count"),
                  mean_mag=("magnitude", "mean"),
                  last_t=("time", "max"))
    if "depth_km" in df_win.columns:
        agg["mean_dep"] = grp["depth_km"].mean()
    else:
        agg["mean_dep"] = np.nan

    rows = agg.index.get_level_values("row").to_numpy(int)
    cols = agg.index.get_level_values("col").to_numpy(int)

    feat[rows, cols, 0] = agg["count"].to_numpy(np.float32)
    feat[rows, cols, 1] = agg["mean_mag"].to_numpy(np.float32)
    feat[rows, cols, 3] = agg["mean_dep"].to_numpy(np.float32)
    feat[rows, cols, 4] = ((win_end - agg["last_t"])
                           .dt.total_seconds().to_numpy(np.float32) / 86_400.0)

    for (r, c), sub in grp:
        feat[r, c, 2] = b_value(sub["magnitude"].to_numpy())

    return feat


# ---------------------------------------------------------------------------
# 5. Build full tensor
# ---------------------------------------------------------------------------

def build_tensor(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    windows = make_windows(df["time"].min(), df["time"].max())
    T       = len(windows)
    tensor  = np.full((T, N_LAT, N_LON, 5), np.nan, dtype=np.float32)
    tensor[:, :, :, 0] = 0.0

    times = df["time"].values

    for idx, win_start, win_end in windows:
        lo = np.searchsorted(times, np.datetime64(win_start), "left")
        hi = np.searchsorted(times, np.datetime64(win_end),   "left")
        tensor[idx] = extract_features(df.iloc[lo:hi], win_end)
        if (idx + 1) % 50 == 0:
            log.info("  … window %d / %d", idx + 1, T)

    log.info("Raw tensor shape: %s", tensor.shape)
    return tensor, windows


# ---------------------------------------------------------------------------
# 6. Per-cell normalisation (fit on training windows only)
# ---------------------------------------------------------------------------

def normalise(tensor: np.ndarray, train_idx: list[int]):
    out = tensor.copy()

    # log1p on event_count (channel 0) — reduces skew
    out[:, :, :, 0] = np.log1p(out[:, :, :, 0].clip(min=0.0))

    # Impute NaN with per-channel training mean
    train_sl = out[train_idx]
    for c in range(5):
        mu = float(np.nanmean(train_sl[:, :, :, c]))
        if np.isnan(mu):
            mu = 0.0
        out[:, :, :, c] = np.where(np.isnan(out[:, :, :, c]), mu, out[:, :, :, c])

    # Per-cell min/max from training windows
    train_sl  = out[train_idx]
    cell_min  = train_sl.min(axis=0)            # (H, W, C)
    cell_max  = train_sl.max(axis=0)
    cell_scale = cell_max - cell_min
    cell_scale[cell_scale == 0.0] = 1.0         # constant cells → stay 0

    out = ((out - cell_min) / cell_scale).clip(0.0, 1.0)
    log.info("Normalised — train mean/channel: %s",
             np.round(out[train_idx].mean(axis=(0, 1, 2)), 3).tolist())
    return out.astype(np.float32), cell_min.astype(np.float32), cell_scale.astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Save HDF5
# ---------------------------------------------------------------------------

def save_hdf5(path, tensor, windows, lat_edges, lon_edges,
              splits, cell_min, cell_scale) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(compression="gzip", compression_opts=4)

    lat_c = ((lat_edges[:-1] + lat_edges[1:]) / 2).astype(np.float32)
    lon_c = ((lon_edges[:-1] + lon_edges[1:]) / 2).astype(np.float32)

    with h5py.File(path, "w") as fh:
        fh.create_dataset("features",      data=tensor,     **opts)
        fh.create_dataset("cell_min",      data=cell_min,   **opts)
        fh.create_dataset("cell_scale",    data=cell_scale, **opts)
        fh.create_dataset("lat_centers",   data=lat_c)
        fh.create_dataset("lon_centers",   data=lon_c)
        fh.create_dataset("feature_names",
                          data=np.array(FEATURE_NAMES, dtype=h5py.string_dtype()))
        fh.create_dataset("window_starts",
                          data=np.array([str(w[1].date()) for w in windows],
                                        dtype=h5py.string_dtype()))
        fh.create_dataset("window_ends",
                          data=np.array([str(w[2].date()) for w in windows],
                                        dtype=h5py.string_dtype()))
        grp = fh.create_group("splits")
        for name, idx in splits.items():
            grp.create_dataset(name, data=np.array(idx, dtype=np.int32))

    log.info("Saved → %s  shape=%s", path, tensor.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_and_clean(RAW_CSV)
    df, lat_edges, lon_edges = assign_grid_cells(df)

    tensor_raw, windows = build_tensor(df)
    T = len(windows)

    n_train = int(T * TRAIN_FRAC)
    n_val   = int(T * VAL_FRAC)
    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, T))
    log.info("Split → train=%d  val=%d  test=%d", n_train, n_val, T - n_train - n_val)

    tensor, cell_min, cell_scale = normalise(tensor_raw, train_idx)

    save_hdf5(OUTPUT_H5, tensor, windows, lat_edges, lon_edges,
              {"train": train_idx, "val": val_idx, "test": test_idx},
              cell_min, cell_scale)

    log.info("=== Done — per-channel stats (full tensor) ===")
    for c, name in enumerate(FEATURE_NAMES):
        ch = tensor[:, :, :, c]
        log.info("  %-30s  mean=%.3f  std=%.3f", name, ch.mean(), ch.std())


if __name__ == "__main__":
    main()