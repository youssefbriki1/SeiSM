#!/usr/bin/env python3
"""
SafeNet Feature Engineering Pipeline (rewrite from scratch).

Reads raw_data.csv and produces a pickle file matching
eqs_and_png_data_for_eval_10y_in_11_16.pickle.

Output structure:
  {'eq_data': [array(10,86,282) for each target_year in 2011..2020],
   'png': [...]}

Key findings from reverse-engineering the reference:
  - Features for history year Y use data from year Y itself
  - Normalization: per-column min-max across ALL regions & years
  - Output dtype: float32
  - Extraction date: Nov 16 (or nearest after)
  - Patch 0 = general map (all data), patches 1-85 = regions from png_list
"""

from pathlib import Path
import pandas as pd
import numpy as np
from lunardate import LunarDate
import pickle
import json
import os
import sys
import image_process

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def solar_to_lunar_day(year, month, day):
    """Convert a solar date to its corresponding lunar day (1-30)."""
    try:
        lunar = LunarDate.fromSolarDate(year, month, day)
        return lunar.day
    except Exception:
        return 15  # fallback

def lunar_month_length(year, month, day):
    """Number of days in the lunar month for a given solar date."""
    try:
        lunar = LunarDate.fromSolarDate(year, month, day)
        try:
            LunarDate(lunar.year, lunar.month, 30)
            return 30
        except ValueError:
            return 29
    except Exception:
        return 30

def normalize_lunar(lunar_day, N=30):
    """sin(x/N * 2*pi) + 2, where N is the lunar month length (29 or 30)."""
    return np.sin(lunar_day / N * 2 * np.pi) + 2

def compute_b_a(magnitudes):
    """Gutenberg-Richter b-value and a-value (Aki-Utsu MLE).
    Returns (b, a) or (0, 0) if insufficient data."""
    if len(magnitudes) < 2:
        return 0.0, 0.0
    m_mean = np.mean(magnitudes)
    m_min = np.min(magnitudes)
    if m_mean == m_min:
        return 0.0, 0.0
    b = np.log10(np.e) / (m_mean - m_min)
    a = np.log10(len(magnitudes)) + b * m_min
    return b, a


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (for one region-year slice)
# ═══════════════════════════════════════════════════════════════════════

def compute_282_features(events_year, events_before, extraction_date):
    """
    Compute 282 features for a single (region, year) pair.

    Args:
        events_year: DataFrame of events in the feature year, sorted by date_time
        events_before: DataFrame of the latest ~100 events up to extraction_date
                       (used for "latest 100" features)
        extraction_date: The extraction date (pd.Timestamp) for this year

    Returns:
        np.array of shape (282,) with float64 values
    """
    features = np.zeros(282)

    # ── Prepare year data ──
    if len(events_year) == 0:
        return features

    year_mags = events_year['magnitude'].values
    year_months = events_year['_month'].values
    year_depths = events_year['depth'].values
    year_dates = events_year['_onlydate_dt'].values
    year_lunar = events_year['_lunar_day'].values if '_lunar_day' in events_year.columns else None
    year_lunar_N = events_year['_lunar_N'].values if '_lunar_N' in events_year.columns else None

    # ───────────────────────────────────────────────────────────────────
    # Cols 0-64: Top 5 magnitudes (year + 12 months)
    # ───────────────────────────────────────────────────────────────────
    def top5_mags(mags):
        if len(mags) == 0:
            return np.zeros(5)
        sorted_m = np.sort(mags)[::-1][:5]
        if len(sorted_m) < 5:
            sorted_m = np.pad(sorted_m, (0, 5 - len(sorted_m)))
        return sorted_m

    # Yearly top 5
    features[0:5] = top5_mags(year_mags)

    # Monthly top 5
    for m in range(1, 13):
        mask = year_months == m
        features[5 + (m-1)*5 : 5 + m*5] = top5_mags(year_mags[mask])

    # ───────────────────────────────────────────────────────────────────
    # Cols 65-129: Lunar date features (normalized) for top 5 mags
    # ───────────────────────────────────────────────────────────────────
    def top5_lunar(mags, lunar_days, lunar_Ns=None):
        if len(mags) == 0:
            return np.zeros(5)
        idx = np.argsort(mags)[::-1][:5]
        ld = lunar_days[idx]
        if lunar_Ns is not None:
            ns = lunar_Ns[idx]
            normalized = np.array([normalize_lunar(d, n) for d, n in zip(ld, ns)])
        else:
            normalized = np.array([normalize_lunar(d) for d in ld])
        if len(normalized) < 5:
            normalized = np.pad(normalized, (0, 5 - len(normalized)))
        return normalized

    if year_lunar is not None:
        features[65:70] = top5_lunar(year_mags, year_lunar, year_lunar_N)
        for m in range(1, 13):
            mask = year_months == m
            features[70 + (m-1)*5 : 70 + m*5] = top5_lunar(
                year_mags[mask], year_lunar[mask],
                year_lunar_N[mask] if year_lunar_N is not None else None
            )

    # ───────────────────────────────────────────────────────────────────
    # Cols 130-177: Magnitude counts per month (4 bins × 12 months)
    # ───────────────────────────────────────────────────────────────────
    for m in range(1, 13):
        mask = year_months == m
        mag = year_mags[mask]
        base = 130 + (m-1)*4
        features[base]     = ((mag >= 0) & (mag <= 3)).sum()
        features[base + 1] = ((mag > 3) & (mag <= 5)).sum()
        features[base + 2] = ((mag > 5) & (mag < 7)).sum()
        features[base + 3] = (mag >= 7).sum()

    # ───────────────────────────────────────────────────────────────────
    # Cols 178-201: Depth counts per month (2 bins × 12 months)
    # ───────────────────────────────────────────────────────────────────
    for m in range(1, 13):
        mask = year_months == m
        depth = year_depths[mask]
        base = 178 + (m-1)*2
        features[base]     = (depth > 70).sum()
        features[base + 1] = (depth <= 70).sum()

    # ───────────────────────────────────────────────────────────────────
    # Cols 202-215: b-value (year + 12 months + latest 100)
    # Cols 216-229: a-value (year + 12 months + latest 100)
    # ───────────────────────────────────────────────────────────────────
    b_yr, a_yr = compute_b_a(year_mags)
    features[202] = b_yr
    features[216] = a_yr

    for m in range(1, 13):
        mask = year_months == m
        b_m, a_m = compute_b_a(year_mags[mask])
        features[202 + m] = b_m
        features[216 + m] = a_m

    # Latest 100 events
    latest_mags = events_before['magnitude'].values[-100:] if len(events_before) > 0 else np.array([])
    b_100, a_100 = compute_b_a(latest_mags)
    features[215] = b_100   # col 215 = b for latest 100
    features[229] = a_100   # col 229 = a for latest 100

    # ───────────────────────────────────────────────────────────────────
    # Cols 230-243: Mean magnitude (year + 12 months + latest 100)
    # ───────────────────────────────────────────────────────────────────
    features[230] = np.mean(year_mags) if len(year_mags) > 0 else 0.0
    for m in range(1, 13):
        mask = year_months == m
        mag_m = year_mags[mask]
        features[230 + m] = np.mean(mag_m) if len(mag_m) > 0 else 0.0
    features[243] = np.mean(latest_mags) if len(latest_mags) > 0 else 0.0

    # ───────────────────────────────────────────────────────────────────
    # Cols 244-256: Std deviation (year + 12 months) — 13 cols, no latest 100
    # ───────────────────────────────────────────────────────────────────
    features[244] = np.std(year_mags) if len(year_mags) > 1 else 0.0
    for m in range(1, 13):
        mask = year_months == m
        mag_m = year_mags[mask]
        features[244 + m] = np.std(mag_m) if len(mag_m) > 1 else 0.0

    # ───────────────────────────────────────────────────────────────────
    # Cols 257-269: P(M >= 6) (year + 12 months) — 13 cols
    # ───────────────────────────────────────────────────────────────────
    features[257] = (year_mags >= 6).sum() / len(year_mags) if len(year_mags) > 0 else 0.0
    for m in range(1, 13):
        mask = year_months == m
        mag_m = year_mags[mask]
        features[257 + m] = (mag_m >= 6).sum() / len(mag_m) if len(mag_m) > 0 else 0.0

    # ───────────────────────────────────────────────────────────────────
    # Cols 270-273: Latest 100 event features
    # ───────────────────────────────────────────────────────────────────
    # Col 270: a/b ratio
    features[270] = a_100 / b_100 if b_100 != 0 else 0.0

    # Col 271: MSE of G-R model for latest 100
    if len(latest_mags) >= 2 and b_100 != 0:
        unique_m = np.sort(np.unique(latest_mags))
        mse_sum = 0.0
        for mv in unique_m:
            actual_n = (latest_mags >= mv).sum()
            pred_log_n = a_100 - b_100 * mv
            actual_log_n = np.log10(actual_n) if actual_n > 0 else 0.0
            mse_sum += (actual_log_n - pred_log_n) ** 2
        features[271] = mse_sum / len(unique_m)
    else:
        features[271] = 0.0

    # Col 272: Time diff between current event and 100th prior event (seconds)
    if len(events_before) > 0:
        latest_dts = events_before['_datetime'].values
        current_dt = latest_dts[-1] if len(latest_dts) > 0 else extraction_date
        ref_idx = max(0, len(latest_dts) - 100)
        ref_dt = latest_dts[ref_idx]
        features[272] = (current_dt - ref_dt) / np.timedelta64(1, 's')

    # Col 273: RMS earthquake energy of latest 100
    if len(latest_mags) > 0:
        energies = np.power(10.0, 1.5 * latest_mags + 4.8)
        features[273] = np.sqrt(np.mean(energies ** 2))

    # ───────────────────────────────────────────────────────────────────
    # Cols 274-277: Mean interval time for 4 mag ranges (latest 100)
    # Cols 278-281: SD/mean interval ratio for 4 mag ranges (latest 100)
    # ───────────────────────────────────────────────────────────────────
    if len(events_before) > 0:
        latest_ev = events_before.iloc[-100:]
        lt_mags = latest_ev['magnitude'].values
        lt_dts = latest_ev['_datetime'].values

        bins = [(0, 3, True), (3, 5, False), (5, 7, False), (7, None, False)]
        for i, (lo, hi, inclusive_lo) in enumerate(bins):
            if hi is not None:
                if inclusive_lo:
                    mask = (lt_mags >= lo) & (lt_mags <= hi)
                else:
                    mask = (lt_mags > lo) & (lt_mags <= hi)
            else:
                mask = lt_mags >= lo

            bin_dts = lt_dts[mask]
            if len(bin_dts) >= 2:
                diffs = np.diff(bin_dts).astype('timedelta64[s]').astype(float)
                mean_iv = np.mean(diffs)
                std_iv = np.std(diffs)
                features[274 + i] = mean_iv
                features[278 + i] = std_iv / mean_iv if mean_iv != 0 else 0.0

    return features


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def load_raw_data(csv_path):
    """Load and prepare raw data with parsed dates and lunar days."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['_datetime'] = pd.to_datetime(df['date_time'])
    df['_onlydate_dt'] = pd.to_datetime(df['onlydate'])
    df['_year'] = df['_onlydate_dt'].dt.year
    df['_month'] = df['_onlydate_dt'].dt.month
    df['_day'] = df['_onlydate_dt'].dt.day

    # Sort by datetime
    df = df.sort_values('_datetime').reset_index(drop=True)

    # Compute lunar days and month lengths
    print("Computing lunar days...")
    df['_lunar_day'] = df.apply(
        lambda r: solar_to_lunar_day(r['_year'], r['_month'], r['_day']),
        axis=1
    )
    df['_lunar_N'] = df.apply(
        lambda r: lunar_month_length(r['_year'], r['_month'], r['_day']),
        axis=1
    )
    print(f"Loaded {len(df)} events, years {df['_year'].min()}-{df['_year'].max()}")
    return df


def load_region_mapping(csv_path):
    """Load the region mapping from png_list_to_patchxy.csv.
    Returns a list of 85 region strings like '(3, 7)'."""
    patch_df = pd.read_csv(csv_path)
    regions = []
    for _, row in patch_df.iterrows():
        x, y = int(row['x']), int(row['y'])
        regions.append(f"({x}, {y})")
    return regions


def find_extraction_event(events_year, year):
    """Find the event on or after Nov 16 of the given year.
    Returns the index into events_year, or the last event if none after Nov 16."""
    target = pd.Timestamp(year=year, month=11, day=16)
    candidates = events_year[events_year['_onlydate_dt'] >= target]
    if len(candidates) > 0:
        return candidates.index[0]
    elif len(events_year) > 0:
        return events_year.index[-1]
    else:
        return None


def compute_features_for_region(df_all, region_filter, years, progress_label=""):
    """
    Compute features for one region across multiple years.

    The feature window for each year Y is the 12 months ending at Nov 16:
      Nov 17 of Y-1  to  Nov 16 of Y  (inclusive)

    The "monthly" features use the calendar month within this window
    (but only events within the 12-month window are included).

    Args:
        df_all: Full raw DataFrame (sorted by datetime)
        region_filter: Region string like '(3, 7)' or None for general map
        years: List of years to compute features for

    Returns:
        dict: {year: np.array(282)} with raw (un-normalized) features
    """
    # Filter by region
    if region_filter is not None:
        df_region = df_all[df_all['region'] == region_filter].copy()
    else:
        df_region = df_all.copy()

    df_region = df_region.sort_values('_datetime').reset_index(drop=True)

    results = {}
    for yr in years:
        # 12-month window: Nov 17 of Y-1 to Nov 16 of Y
        window_start = pd.Timestamp(year=yr - 1, month=11, day=17)
        window_end = pd.Timestamp(year=yr, month=11, day=16)

        events_year = df_region[
            (df_region['_onlydate_dt'] >= window_start) &
            (df_region['_onlydate_dt'] <= window_end)
        ]

        if len(events_year) == 0:
            results[yr] = np.zeros(282)
            continue

        # Extraction date = Nov 16 of year Y (or nearest event)
        extraction_date = window_end

        # Events before extraction date (for "latest 100" features)
        events_before = df_region[df_region['_onlydate_dt'] <= window_end]

        # Compute features
        results[yr] = compute_282_features(events_year, events_before, extraction_date)

    if progress_label:
        print(f"  {progress_label}: computed features for {len(results)} years")

    return results


def build_pickle(raw_csv, patch_csv, output_pickle, norm_start, target_years=None):
    """
    Build the full pickle file.

    Args:
        raw_csv: Path to raw_data.csv
        patch_csv: Path to png_list_to_patchxy.csv
        output_pickle: Output pickle path
        target_years: List of target years (default: 2011-2020)
    """
    if target_years is None:
        target_years = list(range(2011, 2021))

    # Determine all years we need features for
    # Each year Y uses a window from Nov 17 (Y-1) to Nov 16 (Y)
    min_history = min(target_years) - 9  # oldest history year for the output
    max_history = max(target_years)
    
    # For NORMALIZATION, include as many years as possible.
    # With the 1970-2021 dataset, we can start from 1971
    # (window: Nov 17, 1970 to Nov 16, 1971).
    # norm_start = 1971  # earliest year with a full window in the data

    all_feature_years = list(range(norm_start, max_history + 1))
    output_years = list(range(min_history, max_history + 1))
    print(f"Output feature years: {min_history}-{max_history} ({len(output_years)} years)")
    print(f"Normalization years: {norm_start}-{max_history} ({len(all_feature_years)} years)")
    print(f"Data window spans: Nov {norm_start-1} to Nov {max_history}")

    # Load data
    df = load_raw_data(raw_csv)
    regions = load_region_mapping(patch_csv)
    print(f"Loaded {len(regions)} regions")

    # ── Step 1: Compute raw features for ALL regions and ALL years ──
    print("\n=== Step 1: Computing raw features ===")

    # General map (patch 0)
    print("Computing general map features...")
    general_features = compute_features_for_region(
        df, None, all_feature_years, "General map"
    )

    # Individual regions (patches 1-85)
    region_features = {}
    for i, region_str in enumerate(regions):
        region_features[i] = compute_features_for_region(
            df, region_str, all_feature_years,
            f"Region {i}/{len(regions)-1} ({region_str})"
        )

    # ── Step 2: Collect all feature values to compute normalization params ──
    print("\n=== Step 2: Computing normalization parameters ===")

    # Stack all feature vectors into a big matrix
    all_vectors = []
    for yr in all_feature_years:
        all_vectors.append(general_features.get(yr, np.zeros(282)))
        for i in range(len(regions)):
            all_vectors.append(region_features[i].get(yr, np.zeros(282)))

    all_matrix = np.array(all_vectors)  # shape: (N, 282)
    print(f"Total feature vectors for normalization: {all_matrix.shape[0]}")

    col_min = all_matrix.min(axis=0)
    col_max = all_matrix.max(axis=0)
    col_range = col_max - col_min

    # Save normalization params
    norm_params = {}
    for c in range(282):
        norm_params[c] = {'min': float(col_min[c]), 'max': float(col_max[c])}

    params_path = output_pickle.replace('.pickle', '_norm_params.json')
    with open(params_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved normalization params to {params_path}")

    # ── Step 3: Normalize and assemble ──
    print("\n=== Step 3: Normalizing and assembling pickle ===")

    def normalize(vec):
        result = np.zeros(282)
        for c in range(282):
            if col_range[c] != 0:
                result[c] = (vec[c] - col_min[c]) / col_range[c]
            else:
                result[c] = 0.0
        return result

    eq_data = []
    for target_yr in target_years:
        history_years = list(range(target_yr - 9, target_yr + 1))  # 10 years
        year_array = np.zeros((10, 86, 282), dtype=np.float32)

        for h_idx, h_yr in enumerate(history_years):
            # Patch 0 = general map
            raw_vec = general_features.get(h_yr, np.zeros(282))
            year_array[h_idx, 0, :] = normalize(raw_vec).astype(np.float32)

            # Patches 1-85 = individual regions
            for r_idx in range(len(regions)):
                raw_vec = region_features[r_idx].get(h_yr, np.zeros(282))
                year_array[h_idx, r_idx + 1, :] = normalize(raw_vec).astype(np.float32)

        eq_data.append(year_array)
        print(f"  Target year {target_yr}: assembled (10, 86, 282)")

    #norm_start: 10 yeears before target_years[0]
    png_data = image_process.generate_map(target_years[0] - 9, target_years[-1] + 1)

    # ── Step 4: Save pickle ──
    print(f"\n=== Step 4: Saving to {output_pickle} ===")
    output = {
        'eq_data': eq_data,
        'png': png_data
    }
    with open(output_pickle, 'wb') as f:
        pickle.dump(output, f)
    print("Done!")

    return eq_data


def mag_to_class(max_mag):
    """Convert a maximum magnitude to its earthquake class.
    Class 0: 0 <= M < 5
    Class 1: 5 <= M < 6
    Class 2: 6 <= M < 7
    Class 3: M >= 7
    """
    if max_mag >= 7:
        return 3
    if max_mag >= 6:
        return 2
    if max_mag >= 5:
        return 1
    return 0


def build_labels(raw_csv, patch_csv, output_pickle, target_years):
    """
    Build ground truth labels for each target year.

    For eq_data at target year X, the label is the earthquake magnitude
    class observed in year X+1 (the year being predicted).

    Each label entry is an array of 85 integers (one class per patch,
    excluding patch 0 / general map).

    Args:
        raw_csv: Path to the raw earthquake CSV
        patch_csv: Path to png_list_to_patchxy.csv
        output_pickle: Output pickle path for labels
        target_years: List of feature target years (labels will be for target_year + 1)
    """
    print(f"\n{'='*60}")
    print(f"Building labels for {len(target_years)} target years")
    print(f"Feature years: {min(target_years)}-{max(target_years)}")
    print(f"Label years:   {min(target_years)+1}-{max(target_years)+1}")
    print(f"{'='*60}")

    # Load raw data (only need magnitude, date, region)
    print(f"Loading {raw_csv}...")
    df = pd.read_csv(raw_csv, usecols=['magnitude', 'onlydate', 'region'])
    df['_onlydate_dt'] = pd.to_datetime(df['onlydate'])

    # Load region mapping
    regions = load_region_mapping(patch_csv)
    print(f"Loaded {len(regions)} regions")

    # Pre-group by region for fast lookup
    print("Pre-grouping by region...")
    region_groups = {}
    for region_str in regions:
        rdf = df[df['region'] == region_str][['magnitude', '_onlydate_dt']]
        region_groups[region_str] = rdf

    labels = []
    for target_yr in target_years:
        label_yr = target_yr + 1  # predict the NEXT year

        # Label window: Nov 17 of label_yr-1 to Nov 16 of label_yr
        window_start = pd.Timestamp(year=label_yr - 1, month=11, day=17)
        window_end = pd.Timestamp(year=label_yr, month=11, day=16)

        year_labels = np.zeros(85, dtype=np.int64)

        # Patches 0-84: individual regions (fast lookup via pre-grouped data)
        for r_idx, region_str in enumerate(regions):
            rdf = region_groups[region_str]
            mask = (rdf['_onlydate_dt'] >= window_start) & (rdf['_onlydate_dt'] <= window_end)
            max_mag = rdf.loc[mask, 'magnitude'].max() if mask.sum() > 0 else 0.0
            year_labels[r_idx] = mag_to_class(max_mag)

        labels.append(year_labels)
        dist = {c: int((year_labels == c).sum()) for c in range(4)}

        #Print output map e.g Label year 2019 {0: 63, 1: 19, 2: 4, 3: 0}
        print(f"  Label year {label_yr} (target {target_yr}): {dist}")

    # Save
    with open(output_pickle, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Saved labels to {output_pickle}")

    return labels


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
                                      
    DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data'

    raw_csv = os.path.join(DATA_DIR, 'training_data.csv')
    patch_csv = os.path.join(DATA_DIR, 'png_list_to_patchxy.csv')
    output_pickle = os.path.join(DATA_DIR, 'training_output.pickle')
    norm_start = 1971  # earliest year with a full window in the data
    target_years = list(range(1979, 2011)) #1979 - 2010

    # Check files exist
    for f in [raw_csv, patch_csv]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found in {os.getcwd()}")
            sys.exit(1)

    build_pickle(raw_csv, patch_csv, output_pickle, norm_start, target_years)
    build_labels(os.path.join(DATA_DIR, 'training_data.csv'), patch_csv,
                 os.path.join(DATA_DIR, 'training_labels.pickle'), target_years)

    raw_csv = os.path.join(DATA_DIR, 'testing_data.csv')
    patch_csv = os.path.join(DATA_DIR, 'png_list_to_patchxy.csv')
    output_pickle = os.path.join(DATA_DIR, 'testing_output.pickle') #this is what needs to match eqs_and_png.pickle
    norm_start = 2002 # earliest year with a full window in the data
    target_years = list(range(2011, 2021)) #2011 - 2020

    # Check files exist
    for f in [raw_csv, patch_csv]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found in {os.getcwd()}")
            sys.exit(1)

    build_pickle(raw_csv, patch_csv, output_pickle, norm_start, target_years)
    build_labels(os.path.join(DATA_DIR, 'testing_data.csv'), patch_csv,
                 os.path.join(DATA_DIR, 'testing_labels.pickle'), target_years)
