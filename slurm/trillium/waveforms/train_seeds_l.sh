#!/bin/bash
#SBATCH --job-name=wave_seeds_l
#SBATCH --array=0-14
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/seeds_%x-%A_%a.out
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/seeds_%x-%A_%a.err
#SBATCH --export=NONE

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
MAIN_PY=$PROJECT_ROOT/src/waveform_train.py
SIZE=l

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
    SIZE_ARGS="--d_model 512 --d_state 256 --n_layers 8"
    ;;
  bi_lstm)
    SIZE_ARGS="--lstm_hidden_size 512 --lstm_layers 4"
    ;;
  transformer)
    SIZE_ARGS="--d_model 512 --tf_nhead 16 --tf_layers 8 --tf_dim_feedforward 2048"
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
