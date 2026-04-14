#!/bin/bash
#SBATCH --job-name=train_all_waveforms
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/%x-%j.err
#SBATCH --export=NONE          

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
VENV_PY=$PROJECT_ROOT/env/py1013/bin/python
MAIN_PY=$PROJECT_ROOT/src/waveform_train.py
LOG_DIR=$PROJECT_ROOT/slurm_logs

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/.pycache" "$PROJECT_ROOT/wandb"
mkdir -p "$PROJECT_ROOT/.cache/huggingface"

echo "======================================"
echo "Starting Training: Mamba2"
echo "======================================"
srun "$VENV_PY" "$MAIN_PY" \
  --model_type mamba2 \
  --wandb_entity ift3710-ai-slop \
  --wandb_project quake-wave-mamba2 \
  --wandb_run_name wave_mamba2_baseline \
  --save_path "$PROJECT_ROOT/checkpoints/best_mamba2_waveform.pth" \
  --epochs 30 \
  --wandb_mode offline

echo "======================================"
echo "Starting Training: LSTM"
echo "======================================"
srun "$VENV_PY" "$MAIN_PY" \
  --model_type lstm \
  --lstm_hidden_size 128 \
  --lstm_layers 2 \
  --lstm_dropout 0.2 \
  --wandb_entity ift3710-ai-slop \
  --wandb_project quake-wave-mamba2 \
  --wandb_run_name wave_lstm_baseline \
  --save_path "$PROJECT_ROOT/checkpoints/best_lstm_waveform.pth" \
  --epochs 30 \
  --wandb_mode offline

echo "======================================"
echo "Starting Training: Transformer"
echo "======================================"
srun "$VENV_PY" "$MAIN_PY" \
  --model_type transformer \
  --d_model 128 \
  --tf_layers 4 \
  --tf_nhead 8 \
  --tf_dim_feedforward 512 \
  --tf_dropout 0.2 \
  --wandb_entity ift3710-ai-slop \
  --wandb_project quake-wave-mamba2 \
  --wandb_run_name wave_transformer_baseline \
  --save_path "$PROJECT_ROOT/checkpoints/best_transformer_waveform.pth" \
  --epochs 30 \
  --wandb_mode offline

echo "======================================"
echo "All Trainings Completed!"
echo "======================================"
