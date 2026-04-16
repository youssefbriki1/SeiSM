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

<<<<<<< HEAD:src/data-processing/california/run_preprocessing.sh
# Attempt to load Compute Canada dependencies if available to populate $EBROOTPROJ
if command -v module &> /dev/null; then
    module load gcc arrow/22.0.0
fi

=======
source /scratch/brikiyou/ift3710/env/py1013/bin/activate
>>>>>>> ba2b263 (whack updae):src/data-processing/california/run_pre_processing.sh
# Step 1: Download CEED dataset if missing
if [ ! -f "../../../data/california/events.csv" ]; then
    echo "[Step 1] events.csv not found. Downloading CEED dataset..."
<<<<<<< HEAD:src/data-processing/california/run_preprocessing.sh
    python3 ceed_loader.py --catalog_path ../../../data/california/catalog.parquet --base_path ../../../data/california/
=======
    python3 ceed_loader.py
>>>>>>> ba2b263 (whack updae):src/data-processing/california/run_pre_processing.sh
    echo "[Step 1] Download complete."
else
    echo "[Step 1] events.csv already exists. Skipping download."
fi

echo ""

# Step 2: Run full pre-processing pipeline
echo "[Step 2] Running full pre-processing pipeline..."
python3 preprocess_full_pipeline.py
echo "[Step 2] Pre-processing complete."

echo ""
echo "=============================================="
echo " Pipeline finished successfully!"
echo "=============================================="