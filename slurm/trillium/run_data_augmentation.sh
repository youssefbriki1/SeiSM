#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.err
#SBATCH --export=NONE          

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
VENV_PY=$PROJECT_ROOT/env/py1013/bin/python
MAIN_PY=$PROJECT_ROOT/src/data-processing/california/ceed_data_augmentation.py
LOG_DIR=$PROJECT_ROOT/slurm_logs

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.pycache" "$PROJECT_ROOT/wandb"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

srun "$VENV_PY" "$MAIN_PY"
