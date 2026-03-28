#!/usr/bin/env python3
"""
SafeNet Feature Engineering Pipeline — CEED / California edition.

Self-contained: downloads CEED events.csv if missing, generates the
California patch grid, adapts columns to SafeNet format, and runs the
full 282-feature engineering pipeline.

Output structure (data/CEED/processed/):
    ceed_training_output.pickle   features: target years 1996-2010
    ceed_training_labels.pickle   labels:   1997-2011
    ceed_testing_output.pickle    features: target years 2011-2020
    ceed_testing_labels.pickle    labels:   2012-2021

Tensor shape per target year: (10, 65, 282)
    10  — history years
    65  — patch 0 (general California map) + 64 regional patches (8x8 grid)
    282 — min-max normalized features (float32)

California bbox:  lon -125 -> -113,  lat 32 -> 42
Patch grid:       8 cols x 8 rows = 64 patches
"""

import pandas as pd
import numpy as np
from lunardate import LunarDate
from huggingface_hub import hf_hub_download
import pickle
import json
import sys
from pathlib import Path
from ceed_maps_pipeline import run_map_pipeline

# ── Paths ────────────────────────────────────────────────────────────────
CEED_DIR   = Path("data/CEED")
OUTPUT_DIR = Path("data/CEED/processed")
EVENTS_CSV = CEED_DIR / "events.csv"
PATCH_CSV  = CEED_DIR / "png_list_to_patchxy_california.csv"

CEED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── California grid config ───────────────────────────────────────────────
BBOX      = (-125, 32, -113, 42)   # xmin, ymin, xmax, ymax
GRID_COLS = 8
GRID_ROWS = 8

# ── Split config ─────────────────────────────────────────────────────────
TRAIN_START = 1987
TRAIN_END   = 2010
TEST_START  = 2002
TEST_END    = 2020


# ═══════════════════════════════════════════════════════════════════════
# STEP 0A: GENERATE PATCH CSV IF MISSING
# ═══════════════════════════════════════════════════════════════════════

def ensure_patch_csv():
    """Generate png_list_to_patchxy_california.csv if not already present."""
    if PATCH_CSV.exists():
        print(f"Found {PATCH_CSV}, skipping generation.")
        return
    print("Generating California patch grid...")
    rows = [{"x": x, "y": y} for y in range(GRID_ROWS) for x in range(GRID_COLS)]
    df = pd.DataFrame(rows)
    df.to_csv(PATCH_CSV, index=False)
    print(f"Saved {len(df)} patches to {PATCH_CSV}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 0C: ADAPT events.csv → SafeNet-compatible DataFrame
# ═══════════════════════════════════════════════════════════════════════

def assign_patch(lat: float, lon: float) -> str:
    """Map (lat, lon) to patch string '(x, y)' in the 8x8 California grid."""
    xmin, ymin, xmax, ymax = BBOX
    x = int((lon - xmin) / (xmax - xmin) * GRID_COLS)
    y = int((ymax - lat) / (ymax - ymin) * GRID_ROWS)
    x = max(0, min(x, GRID_COLS - 1))
    y = max(0, min(y, GRID_ROWS - 1))
    return f"({x}, {y})"


def load_and_adapt(year_start: int, year_end: int) -> pd.DataFrame:
    """
    Read events.csv, add SafeNet columns, filter to year range.

    Adds:
        date_time  — ISO datetime string (from event_time)
        onlydate   — date-only string YYYY-MM-DD
        depth      — renamed from depth_km
        region     — patch string "(x, y)"
    """
    print(f"Loading {EVENTS_CSV}...")
    df = pd.read_csv(EVENTS_CSV)

    dt = pd.to_datetime(df["event_time"])
    df["date_time"] = dt.dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["onlydate"]  = dt.dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"depth_km": "depth"})

    print("Assigning patch regions...")
    df["region"] = df.apply(
        lambda r: assign_patch(r["latitude"], r["longitude"]), axis=1
    )

    years = dt.dt.year
    df = df[(years >= year_start) & (years <= year_end)].copy()
    print(f"Kept {len(df)} events ({year_start}-{year_end})")

    df.to_csv(f'{CEED_DIR}/events_preprocessed_{year_start}_{year_end}.csv', index=False)

    return df


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def solar_to_lunar_day(year, month, day):
    try:
        return LunarDate.fromSolarDate(year, month, day).day
    except Exception:
        return 15


def lunar_month_length(year, month, day):
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
    return np.sin(lunar_day / N * 2 * np.pi) + 2


def compute_b_a(magnitudes):
    if len(magnitudes) < 2:
        return 0.0, 0.0
    m_mean = np.mean(magnitudes)
    m_min  = np.min(magnitudes)
    if m_mean == m_min:
        return 0.0, 0.0
    b = np.log10(np.e) / (m_mean - m_min)
    a = np.log10(len(magnitudes)) + b * m_min
    return b, a


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_282_features(events_year, events_before, extraction_date):
    features = np.zeros(282)

    if len(events_year) == 0:
        return features

    year_mags    = events_year['magnitude'].values
    year_months  = events_year['_month'].values
    year_depths  = events_year['depth'].values
    year_lunar   = events_year['_lunar_day'].values if '_lunar_day' in events_year.columns else None
    year_lunar_N = events_year['_lunar_N'].values   if '_lunar_N'  in events_year.columns else None

    # ── Cols 0-64: Top 5 magnitudes ──────────────────────────────────
    def top5_mags(mags):
        if len(mags) == 0:
            return np.zeros(5)
        sorted_m = np.sort(mags)[::-1][:5]
        if len(sorted_m) < 5:
            sorted_m = np.pad(sorted_m, (0, 5 - len(sorted_m)))
        return sorted_m

    features[0:5] = top5_mags(year_mags)
    for m in range(1, 13):
        mask = year_months == m
        features[5 + (m-1)*5 : 5 + m*5] = top5_mags(year_mags[mask])

    # ── Cols 65-129: Lunar date features ─────────────────────────────
    def top5_lunar(mags, lunar_days, lunar_Ns=None):
        if len(mags) == 0:
            return np.zeros(5)
        idx = np.argsort(mags)[::-1][:5]
        ld  = lunar_days[idx]
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

    # ── Cols 130-177: Magnitude counts per month ─────────────────────
    for m in range(1, 13):
        mask = year_months == m
        mag  = year_mags[mask]
        base = 130 + (m-1)*4
        features[base]     = ((mag >= 0) & (mag <= 3)).sum()
        features[base + 1] = ((mag > 3)  & (mag <= 5)).sum()
        features[base + 2] = ((mag > 5)  & (mag <  7)).sum()
        features[base + 3] = (mag >= 7).sum()

    # ── Cols 178-201: Depth counts per month ─────────────────────────
    for m in range(1, 13):
        mask  = year_months == m
        depth = year_depths[mask]
        base  = 178 + (m-1)*2
        features[base]     = (depth > 70).sum()
        features[base + 1] = (depth <= 70).sum()

    # ── Cols 202-229: b-value and a-value ────────────────────────────
    b_yr, a_yr = compute_b_a(year_mags)
    features[202] = b_yr
    features[216] = a_yr

    for m in range(1, 13):
        mask = year_months == m
        b_m, a_m = compute_b_a(year_mags[mask])
        features[202 + m] = b_m
        features[216 + m] = a_m

    latest_mags = events_before['magnitude'].values[-100:] if len(events_before) > 0 else np.array([])
    b_100, a_100 = compute_b_a(latest_mags)
    features[215] = b_100
    features[229] = a_100

    # ── Cols 230-243: Mean magnitude ─────────────────────────────────
    features[230] = np.mean(year_mags) if len(year_mags) > 0 else 0.0
    for m in range(1, 13):
        mask  = year_months == m
        mag_m = year_mags[mask]
        features[230 + m] = np.mean(mag_m) if len(mag_m) > 0 else 0.0
    features[243] = np.mean(latest_mags) if len(latest_mags) > 0 else 0.0

    # ── Cols 244-256: Std deviation ───────────────────────────────────
    features[244] = np.std(year_mags) if len(year_mags) > 1 else 0.0
    for m in range(1, 13):
        mask  = year_months == m
        mag_m = year_mags[mask]
        features[244 + m] = np.std(mag_m) if len(mag_m) > 1 else 0.0

    # ── Cols 257-269: P(M >= 6) ───────────────────────────────────────
    features[257] = (year_mags >= 6).sum() / len(year_mags) if len(year_mags) > 0 else 0.0
    for m in range(1, 13):
        mask  = year_months == m
        mag_m = year_mags[mask]
        features[257 + m] = (mag_m >= 6).sum() / len(mag_m) if len(mag_m) > 0 else 0.0

    # ── Cols 270-273: Latest 100 summary features ─────────────────────
    features[270] = a_100 / b_100 if b_100 != 0 else 0.0

    if len(latest_mags) >= 2 and b_100 != 0:
        unique_m = np.sort(np.unique(latest_mags))
        mse_sum  = 0.0
        for mv in unique_m:
            actual_n     = (latest_mags >= mv).sum()
            pred_log_n   = a_100 - b_100 * mv
            actual_log_n = np.log10(actual_n) if actual_n > 0 else 0.0
            mse_sum += (actual_log_n - pred_log_n) ** 2
        features[271] = mse_sum / len(unique_m)

    if len(events_before) > 0:
        latest_dts = events_before['_datetime'].values
        current_dt = latest_dts[-1] if len(latest_dts) > 0 else extraction_date
        ref_idx    = max(0, len(latest_dts) - 100)
        features[272] = (current_dt - latest_dts[ref_idx]) / np.timedelta64(1, 's')

    if len(latest_mags) > 0:
        energies      = np.power(10.0, 1.5 * latest_mags + 4.8)
        features[273] = np.sqrt(np.mean(energies ** 2))

    # ── Cols 274-281: Interval time stats ────────────────────────────
    if len(events_before) > 0:
        latest_ev = events_before.iloc[-100:]
        lt_mags   = latest_ev['magnitude'].values
        lt_dts    = latest_ev['_datetime'].values

        bins = [(0, 3, True), (3, 5, False), (5, 7, False), (7, None, False)]
        for i, (lo, hi, inclusive_lo) in enumerate(bins):
            if hi is not None:
                mask = (lt_mags >= lo) & (lt_mags <= hi) if inclusive_lo else (lt_mags > lo) & (lt_mags <= hi)
            else:
                mask = lt_mags >= lo
            bin_dts = lt_dts[mask]
            if len(bin_dts) >= 2:
                diffs   = np.diff(bin_dts).astype('timedelta64[s]').astype(float)
                mean_iv = np.mean(diffs)
                std_iv  = np.std(diffs)
                features[274 + i] = mean_iv
                features[278 + i] = std_iv / mean_iv if mean_iv != 0 else 0.0

    return features


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def prepare_dataframe(df_adapted: pd.DataFrame) -> pd.DataFrame:
    """Add internal helper columns needed by feature computation."""
    df = df_adapted.copy()
    df['_datetime']    = pd.to_datetime(df['date_time'])
    df['_onlydate_dt'] = pd.to_datetime(df['onlydate'])
    df['_year']        = df['_onlydate_dt'].dt.year
    df['_month']       = df['_onlydate_dt'].dt.month
    df['_day']         = df['_onlydate_dt'].dt.day
    df = df.sort_values('_datetime').reset_index(drop=True)

    print("Computing lunar days...")
    df['_lunar_day'] = df.apply(
        lambda r: solar_to_lunar_day(r['_year'], r['_month'], r['_day']), axis=1
    )
    df['_lunar_N'] = df.apply(
        lambda r: lunar_month_length(r['_year'], r['_month'], r['_day']), axis=1
    )
    print(f"Ready: {len(df)} events, years {df['_year'].min()}-{df['_year'].max()}")
    return df


def load_region_mapping() -> list:
    """Load region list from the California patch CSV."""
    patch_df = pd.read_csv(PATCH_CSV)
    return [f"({int(row['x'])}, {int(row['y'])})" for _, row in patch_df.iterrows()]


# ═══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION PER REGION
# ═══════════════════════════════════════════════════════════════════════

def compute_features_for_region(df_all, region_filter, years, progress_label=""):
    df_region = df_all[df_all['region'] == region_filter].copy() if region_filter else df_all.copy()
    df_region = df_region.sort_values('_datetime').reset_index(drop=True)

    results = {}
    for yr in years:
        window_start = pd.Timestamp(year=yr - 1, month=11, day=17)
        window_end   = pd.Timestamp(year=yr,     month=11, day=16)

        events_year = df_region[
            (df_region['_onlydate_dt'] >= window_start) &
            (df_region['_onlydate_dt'] <= window_end)
        ]

        if len(events_year) == 0:
            results[yr] = np.zeros(282)
            continue

        events_before = df_region[df_region['_onlydate_dt'] <= window_end]
        results[yr]   = compute_282_features(events_year, events_before, window_end)

    if progress_label:
        print(f"  {progress_label}: computed features for {len(results)} years")

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def build_pickle(df, output_pickle, norm_start, target_years):
    """
    Build feature pickle from a prepared DataFrame.

    Tensor shape per target year: (10, 65, 282)
        10  — history years
        65  — patch 0 (general California map) + 64 regional patches
        282 — normalized features
    """
    n_total = GRID_COLS * GRID_ROWS + 1   # 64 patches + general map

    max_history       = max(target_years)
    all_feature_years = list(range(norm_start, max_history + 1))

    print(f"Normalization years: {norm_start}-{max_history} ({len(all_feature_years)} years)")
    print(f"Target years:        {min(target_years)}-{max_history}")

    regions = load_region_mapping()
    print(f"Loaded {len(regions)} patches")

    # ── Step 1: Raw features ──────────────────────────────────────────
    print("\n=== Step 1: Computing raw features ===")
    print("Computing general map features...")
    general_features = compute_features_for_region(df, None, all_feature_years, "General map")

    region_features = {}
    for i, region_str in enumerate(regions):
        region_features[i] = compute_features_for_region(
            df, region_str, all_feature_years,
            f"Region {i}/{len(regions)-1} ({region_str})"
        )

    # ── Step 2: Normalization ─────────────────────────────────────────
    print("\n=== Step 2: Computing normalization parameters ===")
    all_vectors = []
    for yr in all_feature_years:
        all_vectors.append(general_features.get(yr, np.zeros(282)))
        for i in range(len(regions)):
            all_vectors.append(region_features[i].get(yr, np.zeros(282)))

    all_matrix = np.array(all_vectors)
    print(f"Total feature vectors for normalization: {all_matrix.shape[0]}")

    col_min   = all_matrix.min(axis=0)
    col_max   = all_matrix.max(axis=0)
    col_range = col_max - col_min

    params_path = str(output_pickle).replace('.pickle', '_norm_params.json')
    with open(params_path, 'w') as f:
        json.dump({c: {'min': float(col_min[c]), 'max': float(col_max[c])} for c in range(282)}, f, indent=2)
    print(f"Saved normalization params to {params_path}")

    # ── Step 3: Assemble ──────────────────────────────────────────────
    print("\n=== Step 3: Normalizing and assembling pickle ===")

    def normalize(vec):
        result = np.zeros(282)
        for c in range(282):
            if col_range[c] != 0:
                result[c] = (vec[c] - col_min[c]) / col_range[c]
        return result

    eq_data = []
    png_data = []
    for target_yr in target_years:
        history_years = list(range(target_yr - 9, target_yr + 1))
        year_array    = np.zeros((10, n_total, 282), dtype=np.float32)

        for h_idx, h_yr in enumerate(history_years):
            year_array[h_idx, 0, :] = normalize(general_features.get(h_yr, np.zeros(282))).astype(np.float32)
            for r_idx in range(len(regions)):
                year_array[h_idx, r_idx + 1, :] = normalize(region_features[r_idx].get(h_yr, np.zeros(282))).astype(np.float32)

        eq_data.append(year_array)
        print(f"  Target year {target_yr}: assembled (10, {n_total}, 282)")

    # Load png data for the full history range (10 years before first target year)
    png_start = min(target_years) - 9
    png_end = max(target_years)
    for yr in range(png_start, png_end + 1):
        png = np.load(f'data/processed/cal_maps/patches_{yr}.npy')
        png = png.transpose(0, 2, 3, 1)  # (N, 5, H, W) → (N, H, W, 5)
        png_data.append(png)
    print(f"  Loaded png data for years {png_start}-{png_end} ({len(png_data)} entries)")


    # ── Step 4: Save ──────────────────────────────────────────────────
    print(f"\n=== Step 4: Saving to {output_pickle} ===")
    with open(output_pickle, 'wb') as f:
        pickle.dump({'eq_data': eq_data, 'png': png_data}, f)
    print("Done!")
    return eq_data


def mag_to_class(max_mag):
    if max_mag >= 7: return 3
    if max_mag >= 6: return 2
    if max_mag >= 5: return 1
    return 0


def build_labels(df, output_pickle, target_years):
    """Build ground truth labels. Each entry is an array of 64 ints (one per patch)."""
    print(f"\n{'='*60}")
    print(f"Building labels: feature years {min(target_years)}-{max(target_years)}, "
          f"label years {min(target_years)+1}-{max(target_years)+1}")
    print(f"{'='*60}")

    regions  = load_region_mapping()
    df_label = df[['magnitude', 'onlydate', 'region']].copy()
    df_label['_onlydate_dt'] = pd.to_datetime(df_label['onlydate'])

    print("Pre-grouping by region...")
    region_groups = {
        r: df_label[df_label['region'] == r][['magnitude', '_onlydate_dt']]
        for r in regions
    }

    labels = []
    for target_yr in target_years:
        label_yr     = target_yr + 1
        window_start = pd.Timestamp(year=label_yr - 1, month=11, day=17)
        window_end   = pd.Timestamp(year=label_yr,     month=11, day=16)

        year_labels = np.zeros(len(regions), dtype=np.int64)
        for r_idx, region_str in enumerate(regions):
            rdf     = region_groups[region_str]
            mask    = (rdf['_onlydate_dt'] >= window_start) & (rdf['_onlydate_dt'] <= window_end)
            max_mag = rdf.loc[mask, 'magnitude'].max() if mask.sum() > 0 else 0.0
            year_labels[r_idx] = mag_to_class(max_mag)

        labels.append(year_labels)
        dist = {c: int((year_labels == c).sum()) for c in range(4)}
        print(f"  Label year {label_yr} (target {target_yr}): {dist}")

    with open(output_pickle, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Saved labels to {output_pickle}")
    return labels


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Step 0: Ensure patch CSV exist ────────────────────────────────
    ensure_patch_csv()

    # ── Training (target years 1996-2010) ─────────────────────────────
    # norm_start=1988: first full Nov17->Nov16 window (CEED starts 1987)
    # target_years start at 1996 to guarantee 10 years of history back to 1987
    print("\n" + "="*60)
    print("TRAINING (target years 1996-2010)")
    print("="*60)
    df_train = prepare_dataframe(load_and_adapt(TRAIN_START, TRAIN_END))
    run_map_pipeline(TRAIN_START, TRAIN_END)
    build_pickle(df_train, str(OUTPUT_DIR / 'ceed_training_output.pickle'),
                 norm_start=1988, target_years=list(range(1996, 2011)))
    build_labels(df_train, str(OUTPUT_DIR / 'ceed_training_labels.pickle'),
                 target_years=list(range(1996, 2011)))

    # ── Testing (target years 2011-2020) ──────────────────────────────
    # norm_start=2002: matches testing data start, same logic as Chinese pipeline
    print("\n" + "="*60)
    print("TESTING (target years 2011-2020)")
    print("="*60)
    df_test = prepare_dataframe(load_and_adapt(TEST_START, TEST_END))
    run_map_pipeline(TEST_START, TEST_END)
    build_pickle(df_test, str(OUTPUT_DIR / 'ceed_testing_output.pickle'),
                 norm_start=2002, target_years=list(range(2011, 2021)))
    build_labels(df_test, str(OUTPUT_DIR / 'ceed_testing_labels.pickle'),
                 target_years=list(range(2011, 2021)))
