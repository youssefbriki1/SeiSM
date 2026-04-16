#!/bin/bash
#SBATCH --job-name=wave_train_array
#SBATCH --array=0-2            
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=4:00:00        
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/scaled_up_%x-%A_%a.out 
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/scaled_up_%x-%A_%a.err
#SBATCH --export=NONE          

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
MAIN_PY=$PROJECT_ROOT/src/waveform_train.py

mkdir -p "$PROJECT_ROOT/slurm_logs" "$PROJECT_ROOT/checkpoints"
source "$PROJECT_ROOT/env/py1013/bin/activate"

MODELS=("mamba2"  "bi_lstm" "transformer")
OPTIMIZERS=("adamw" "adamw"  "adamw")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
OPT=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}

# Changed "baseline" to "scaled"
RUN_NAME="wave_${MODEL}_${OPT}_scaled"
SAVE_PATH="$PROJECT_ROOT/checkpoints/best_${MODEL}_${OPT}_waveform.pth"

echo "======================================"
echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL | Optimizer: $OPT"
echo "======================================"

CMD="python $MAIN_PY --model_type $MODEL --optimizer $OPT --wandb_entity ift3710-ai-slop --wandb_project quake-wave-mamba2 --wandb_run_name $RUN_NAME --save_path $SAVE_PATH --epochs 30 --wandb_mode offline"

# Injected the scaled-up arguments for all models
# if [ "$MODEL" == "lstm" ]; then
#     CMD="$CMD --lstm_hidden_size 256 --lstm_layers 4 --lstm_dropout 0.2 --lstm_bidirectional"
# elif [ "$MODEL" == "transformer" ]; then
#     CMD="$CMD --d_model 256 --tf_layers 8 --tf_nhead 16 --tf_dim_feedforward 1024 --tf_dropout 0.2"
# elif [ "$MODEL" == "mamba2" ]; then
#     CMD="$CMD --d_model 256 --mamba_layers 8 --mamba_d_state 64 --mamba_expand 4 --mamba_dropout 0.2"
# fi

# Run it!
eval $CMD