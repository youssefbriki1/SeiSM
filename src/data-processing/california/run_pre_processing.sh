#!/bin/bash
# ==============================================================
# CEED California Pre-Processing Pipeline
# ==============================================================
# 1. Downloads the CEED dataset (events.csv) if not already present
# 2. Runs the full feature-engineering pipeline
# ==============================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " CEED California Pre-Processing Pipeline"
echo "=============================================="
echo "Working directory: $(pwd)"
echo ""

# Step 1: Download CEED dataset if missing
if [ ! -f "data/CEED/events.csv" ]; then
    echo "[Step 1] events.csv not found. Downloading CEED dataset..."
    uv run ceed_loader.py
    echo "[Step 1] Download complete."
else
    echo "[Step 1] events.csv already exists. Skipping download."
fi

echo ""

# Step 2: Run full pre-processing pipeline
echo "[Step 2] Running full pre-processing pipeline..."
export PROJ_LIB=${EBROOTPROJ}/share/proj
export PROJ_DATA=${EBROOTPROJ}/share/proj
uv run preprocess_full_pipeline.py
echo "[Step 2] Pre-processing complete."

echo ""
echo "=============================================="
echo " Pipeline finished successfully!"
echo "=============================================="
