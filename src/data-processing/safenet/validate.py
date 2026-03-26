#!/usr/bin/env python3
"""
Validate pipeline output against the reference pickle.
Run this AFTER pipeline.py has produced replicated_output.pickle.
"""
import pickle
import numpy as np
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

REF_PICKLE = os.path.join(DATA_DIR, 'maps', 'eqs_and_png_data_for_eval_10y_in_11_16.pickle')
OUR_PICKLE = os.path.join(DATA_DIR, 'data', 'testing_output.pickle')

COL_RANGES = [
    (0,   64,  "Top 5 magnitudes"),
    (65,  129, "Lunar dates"),
    (130, 177, "Mag counts"),
    (178, 201, "Depth counts"),
    (202, 215, "b-value"),
    (216, 229, "a-value"),
    (230, 243, "Mean mag"),
    (244, 256, "Std dev"),
    (257, 269, "P(M>=6)"),
    (270, 270, "a/b ratio"),
    (271, 271, "MSE G-R"),
    (272, 272, "Time diff"),
    (273, 273, "RMS energy"),
    (274, 277, "Mean interval"),
    (278, 281, "SD/mean interval"),
]

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare(ref, ours, label=""):
    ref_eq = ref['eq_data']
    our_eq = ours['eq_data']

    print(f"\n{'='*80}")
    print(f" {label}")
    print(f"{'='*80}")
    print(f"  Ref entries: {len(ref_eq)}, Our entries: {len(our_eq)}")

    for yr_idx in range(min(len(ref_eq), len(our_eq))):
        target_yr = 2011 + yr_idx
        ref_arr = ref_eq[yr_idx]
        our_arr = our_eq[yr_idx]
        
        diff = np.abs(ref_arr.astype(np.float64) - our_arr.astype(np.float64))
        print(f"\n  Target year {target_yr}:")
        print(f"    Overall: max={diff.max():.6f}  mean={diff.mean():.6f}  <1e-3={(diff<1e-3).mean():.3f}")
        
        # Per feature group (across all patches and history years)
        for start, end, name in COL_RANGES:
            g = diff[:, :, start:end+1]
            print(f"    [{start:3d}-{end:3d}] {name:25s}: max={g.max():.6f}  mean={g.mean():.6f}  <1e-2={(g<0.01).mean():.3f}")

    # Detailed look at one test case: target=2012, history[0], patch=0
    print(f"\n{'='*80}")
    print(f" DETAILED: target=2012, history[0], patch=0 (general map)")
    print(f"{'='*80}")
    ref_row = ref_eq[1][0][0].astype(np.float64)
    our_row = our_eq[1][0][0].astype(np.float64)
    diff_row = np.abs(ref_row - our_row)
    
    print(f"  Max diff: {diff_row.max():.6f}")
    print(f"  Mean diff: {diff_row.mean():.6f}")
    
    worst = np.argsort(diff_row)[-15:][::-1]
    print(f"\n  Top 15 worst columns:")
    for col in worst:
        print(f"    col[{col:3d}]: ref={ref_row[col]:.8f}  ours={our_row[col]:.8f}  diff={diff_row[col]:.8f}")

    # Check patch 1 too
    print(f"\n{'='*80}")
    print(f" DETAILED: target=2012, history[0], patch=1 (first region)")
    print(f"{'='*80}")
    ref_row = ref_eq[1][0][1].astype(np.float64)
    our_row = our_eq[1][0][1].astype(np.float64)
    diff_row = np.abs(ref_row - our_row)
    
    print(f"  Max diff: {diff_row.max():.6f}")
    print(f"  Mean diff: {diff_row.mean():.6f}")
    
    worst = np.argsort(diff_row)[-15:][::-1]
    print(f"\n  Top 15 worst columns:")
    for col in worst:
        print(f"    col[{col:3d}]: ref={ref_row[col]:.8f}  ours={our_row[col]:.8f}  diff={diff_row[col]:.8f}")


if __name__ == '__main__':
    print("Loading reference pickle...")
    ref = load_pickle(REF_PICKLE)
    
    if not os.path.exists(OUR_PICKLE):
        print(f"ERROR: {OUR_PICKLE} not found. Run pipeline.py first.")
        exit(1)
    
    print("Loading our pickle...")
    ours = load_pickle(OUR_PICKLE)
    
    compare(ref, ours, "FULL COMPARISON")
