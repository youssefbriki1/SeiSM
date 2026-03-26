#!/usr/bin/env bash
# run_preprocessing.sh — SafeNet data pipeline runner
# Usage: bash run_preprocessing.sh [--validate]

set -e

PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE=false

# ── Parse args ────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --validate) VALIDATE=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── Check scripts exist ────────────────────────────────────────────────
required_scripts=(split_data.py pipeline.py)
$VALIDATE && required_scripts+=(validate.py validate_labeled_data.py)

for script in "${required_scripts[@]}"; do
    [[ -f "$SCRIPT_DIR/$script" ]] || die "$script not found in $SCRIPT_DIR"
done

cd "$SCRIPT_DIR" || die "Cannot cd into $SCRIPT_DIR"

# ── Step 1: Split raw data ────────────────────────────────────────────
log "Step 1/2: Splitting raw data into training/testing CSVs..."
$PYTHON split_data.py || die "split_data.py failed"
log "Done."

# ── Step 2: Feature engineering ───────────────────────────────────────
log "Step 2/2: Building feature pickles + labels (this may take a while)..."
$PYTHON pipeline.py || die "pipeline.py failed"
log "Done."

# ── Step 3 & 4: Validation (optional) ────────────────────────────────
if $VALIDATE; then
    log "Step 3/4: Validating features against reference pickle..."
    $PYTHON validate.py || die "validate.py failed"
    log "Done."

    log "Step 4/4: Validating labels against reference pickle..."
    $PYTHON validate_labeled_data.py || die "validate_labeled_data.py failed"
    log "Done."
else
    log "Skipping validation (pass --validate to enable)."
fi

log "All steps completed successfully."