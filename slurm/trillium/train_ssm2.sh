#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH -p debug 
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.err
#SBATCH --export=NONE          

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14
PROJECT_ROOT=/scratch/brikiyou/ift3710
VENV_PY=$PROJECT_ROOT/env/py1013/bin/python
MAIN_PY=$PROJECT_ROOT/src/main.py
LOG_DIR=$PROJECT_ROOT/slurm_logs

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.pycache" "$PROJECT_ROOT/wandb"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

srun "$VENV_PY" "$MAIN_PY" \
  --wandb_entity ift3710-ai-slop \
  --wandb_project quake-wave-mamba2 \
  --wandb_run_name v1_wave_mamba2_epoch_20 \
  --epochs 20 \
  --wandb_mode offline \
  "$@"
