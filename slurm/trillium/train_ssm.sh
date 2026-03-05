#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.err
#SBATCH --export=NONE          
set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6

# ---- Project paths ----
PROJECT_ROOT=/scratch/brikiyou/ift3710
VENV_PY=$PROJECT_ROOT/.venv/bin/python
DATA_DIR=$PROJECT_ROOT/src/data-processing/data
MAIN_PY=$PROJECT_ROOT/src/main.py
LOG_DIR=$PROJECT_ROOT/slurm_logs

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.pycache" "$PROJECT_ROOT/wandb"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

# ---- Disable Python environment variables injected by CVMFS ----
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX=$PROJECT_ROOT/.pycache

# ---- Set WandB log directory under $SCRATCH ----
export WANDB_DIR=$PROJECT_ROOT/wandb
export WANDB_MODE=offline
unset TRANSFORMERS_CACHE
export HF_HOME=$PROJECT_ROOT/.cache/huggingface

echo "[run.sh] Using: $VENV_PY"
$VENV_PY -V
$VENV_PY -I -c "import sys, site; \
print('prefix =', sys.prefix); \
print('sitepkgs =', [p for p in site.getsitepackages() if 'site-packages' in p])"
# Force check that numpy can be imported
$VENV_PY -I -c "import numpy; print('numpy OK:', numpy.__version__)"

# ---- Execution ----
cd "$PROJECT_ROOT"
echo "[Command] : srun $VENV_PY $MAIN_PY --data_dir $DATA_DIR --wandb_mode offline $*"
srun "$VENV_PY" "$MAIN_PY" \
  --data_dir "$DATA_DIR" \
  --wandb_mode offline \
  "$@"
