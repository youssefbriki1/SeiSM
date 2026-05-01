#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.err
#SBATCH --export=NONE          
set -euo pipefail

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

# ---- Tell Triton / pip to use the module-loaded GCC, not /bin/gcc ----
export CC=$(which gcc)
export CXX=$(which g++)
export HOME=${HOME:-/scratch/brikiyou}
echo "[env] CC=$CC  CXX=$CXX"

# ---- Clear stale Triton JIT cache (may have cached /bin/gcc path) ----
rm -rf /scratch/brikiyou/triton_cache 2>/dev/null || true
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

# ---- Project paths ----
PROJECT_ROOT=/scratch/brikiyou/ift3710
VENV_PY=$PROJECT_ROOT/env/py1013/bin/python
DATA_DIR=$PROJECT_ROOT/src/data-processing/california/data/CEED/processed
MAIN_PY=$PROJECT_ROOT/src/main_mutimodal.py
LOG_DIR=$PROJECT_ROOT/slurm_logs

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.pycache" "$PROJECT_ROOT/wandb"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

# ---- Allow ComputeCanada LMOD to manage Python environment variables ----
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX=$PROJECT_ROOT/.pycache

# ---- Set WandB log directory under $PROJECT_ROOT ----
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
echo "[Command] : srun $VENV_PY $MAIN_PY --model safenet_ssm --use_focal_loss --data_dir $DATA_DIR --wandb_mode offline $*"
srun --export=ALL "$VENV_PY" "$MAIN_PY" \
  --model safenet_ssm \
  --data_dir "$DATA_DIR" \
  --use_focal_loss \
  --wandb_mode offline \
  --train_target_year_start 1987 \
  --epochs 500 \
  --lr 1e-5 \
  --weight_decay 1e-3 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --focal_gamma 3.0 \
  --focal_alpha 1.0 4.0 15.0 78.0 \
  "$@"