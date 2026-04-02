#!/bin/bash
#SBATCH --job-name=eval_ssm
#SBATCH --output=slurm_logs/eval_ssm-%j.out
#SBATCH --error=slurm_logs/eval_ssm-%j.err
#SBATCH --time=00:20:00
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
VENV_PY=$PROJECT_ROOT/env/py1013/bin/python
MAIN_PY=$PROJECT_ROOT/src/evaluate_waveform.py
LOG_DIR=$PROJECT_ROOT/slurm_logs

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.pycache" "$PROJECT_ROOT/wandb"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

srun "$VENV_PY" "$MAIN_PY"