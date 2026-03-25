#!/usr/bin/env bash
# run_preprocessing.sh — SafeNet data pipeline runner
# Usage: bash run_preprocessing.sh

set -e  # exit immediately on error

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── Check scripts exist ────────────────────────────────────────────────
for script in split_data.py pipeline.py validate.py validate_labeled_data.py; do
    [[ -f "$SCRIPT_DIR/$script" ]] || die "$script not found in $SCRIPT_DIR"
done

# ── Step 1: Split raw data ─────────────────────────────────────────────
log "Step 1/4: Splitting raw data into training/testing CSVs..."
$PYTHON "$SCRIPT_DIR/split_data.py" || die "split_data.py failed"
log "Done."

# ── Step 2: Feature engineering ───────────────────────────────────────
log "Step 2/4: Building feature pickles + labels (this may take a while)..."
$PYTHON "$SCRIPT_DIR/pipeline.py" || die "pipeline.py failed"
log "Done."

# ── Step 3: Validate features ─────────────────────────────────────────
log "Step 3/4: Validating features against reference pickle..."
$PYTHON "$SCRIPT_DIR/validate.py" || die "validate.py failed"
log "Done."

# ── Step 4: Validate labels ───────────────────────────────────────────
log "Step 4/4: Validating labels against reference pickle..."
$PYTHON "$SCRIPT_DIR/validate_labeled_data.py" || die "validate_labeled_data.py failed"
log "Done."

log "All steps completed successfully."