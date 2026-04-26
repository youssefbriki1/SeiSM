#!/bin/bash
#SBATCH --job-name=wave_seeds_xs
#SBATCH --array=0-14
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/seeds_%x-%A_%a.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/seeds_%x-%A_%a.err
#SBATCH --export=NONE

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
MAIN_PY=$PROJECT_ROOT/src/waveform_train.py
SIZE=xs

mkdir -p "$PROJECT_ROOT/slurm_logs" "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/results"
source "$PROJECT_ROOT/env/py1013/bin/activate"

MODELS=("mamba2" "bi_lstm" "transformer")
SEEDS=(42 1 2 3 4)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

case $MODEL in
  mamba2)
    SIZE_ARGS="--d_model 64 --d_state 32 --n_layers 2"
    ;;
  bi_lstm)
    SIZE_ARGS="--lstm_hidden_size 64 --lstm_layers 1"
    ;;
  transformer)
    SIZE_ARGS="--d_model 64 --tf_nhead 4 --tf_layers 2 --tf_dim_feedforward 256"
    ;;
esac

RUN_NAME="wave_${MODEL}_${SIZE}_seed${SEED}"
SAVE_PATH="$PROJECT_ROOT/checkpoints/best_${MODEL}_${SIZE}_seed${SEED}.pth"
JSON_PATH="$PROJECT_ROOT/results/{model}_${SIZE}_seed{seed}.json"

echo "======================================"
echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL | Size: $SIZE | Seed: $SEED"
echo "======================================"

python $MAIN_PY \
    --model_type $MODEL \
    --optimizer adamw \
    --seed $SEED \
    --wandb_entity ift3710-ai-slop \
    --wandb_project quake-wave-mamba2 \
    --wandb_run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --json_log_path "$JSON_PATH" \
    --epochs 30 \
    --wandb_mode offline \
    $SIZE_ARGS
