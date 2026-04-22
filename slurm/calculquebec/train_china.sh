#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --partition=nodegpupool
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --output=/project/60004/fauverick/ift3710/slurm_logs/%x-%j.out
#SBATCH --error=/project/60004/fauverick/ift3710/slurm_logs/%x-%j.err
#SBATCH --export=NONE          
set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load arrow/22.0.0

# ---- Tell Triton / pip to use the module-loaded GCC, not /bin/gcc ----
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME=${EBROOTCUDA:-$CUDA_HOME}
export HOME=${HOME:-/project/60004/fauverick}
echo "[env] CC=$CC  CXX=$CXX  CUDA_HOME=$CUDA_HOME"

# ---- Clear stale Triton JIT cache (may have cached /bin/gcc path) ----
rm -rf $HOME/.triton/cache 2>/dev/null || true
rm -rf /tmp/triton_* 2>/dev/null || true

# ---- Project paths ----
PROJECT_ROOT=/project/60004/fauverick/ift3710
VENV_PY=$PROJECT_ROOT/.venv/bin/python
DATA_DIR=$PROJECT_ROOT/src/data-processing/safenet/data
MAIN_PY=$PROJECT_ROOT/src/main_mutimodal.py
LOG_DIR=$PROJECT_ROOT/slurm_logs
SAVE_PATH=$PROJECT_ROOT/checkpoints/best_safenet_full_china.pth

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
echo "[Command] : srun $VENV_PY $MAIN_PY --use_focal_loss  --data_dir $DATA_DIR --wandb_mode offline $*"
srun --export=ALL "$VENV_PY" "$MAIN_PY" \
  --data_dir "$DATA_DIR" \
  --save_path "$SAVE_PATH" \
  --use_focal_loss \
  --wandb_mode offline \
  --train_target_year_start 1979 \
  --epochs 300 \
  --lr 1e-5 \
  --weight_decay 1e-3 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --focal_gamma 3.0 \
  --focal_alpha 1.0 4.0 15.0 78.0 \
  --train_features_file "training_output.pickle" \
  --train_labels_file "training_labels.pickle" \
  --val_features_file "testing_output.pickle" \
  --val_labels_file "testing_labels.pickle" \
  --test_features_file "testing_output.pickle" \
  --test_labels_file "testing_labels.pickle"
  "$@"
