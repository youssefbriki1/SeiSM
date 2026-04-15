#!/bin/bash
#SBATCH --job-name=wave_train_array
#SBATCH --array=0-5%2    
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=4:00:00        
#SBATCH --output=/scratch/brikiyou/ift3710/slurm_logs/%x-%A_%a.out 
#SBATCH --error=/scratch/brikiyou/ift3710/slurm_logs/%x-%A_%a.err
#SBATCH --export=NONE          

module load StdEnv/2023 gcc/12.3 python/3.10 arrow/14

mkdir -p /scratch/brikiyou/triton_cache
export TRITON_CACHE_DIR=/scratch/brikiyou/triton_cache

PROJECT_ROOT=/scratch/brikiyou/ift3710
MAIN_PY=$PROJECT_ROOT/src/waveform_train.py

mkdir -p "$PROJECT_ROOT/slurm_logs" "$PROJECT_ROOT/checkpoints"
source "$PROJECT_ROOT/env/py1013/bin/activate"

MODELS=("mamba2" "mamba2" "lstm" "lstm" "transformer" "transformer")
OPTIMIZERS=("adamw" "muon" "adamw" "muon" "adamw" "muon")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
OPT=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}

RUN_NAME="wave_${MODEL}_${OPT}_baseline"
SAVE_PATH="$PROJECT_ROOT/checkpoints/best_${MODEL}_${OPT}_waveform.pth"

echo "======================================"
echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL | Optimizer: $OPT"
echo "======================================"

CMD="python $MAIN_PY --model_type $MODEL --optimizer $OPT --wandb_entity ift3710-ai-slop --wandb_project quake-wave-mamba2 --wandb_run_name $RUN_NAME --save_path $SAVE_PATH --epochs 30 --wandb_mode offline"

if [ "$MODEL" == "lstm" ]; then
    CMD="$CMD --lstm_hidden_size 128 --lstm_layers 2 --lstm_dropout 0.2"
elif [ "$MODEL" == "transformer" ]; then
    CMD="$CMD --d_model 128 --tf_layers 4 --tf_nhead 8 --tf_dim_feedforward 512 --tf_dropout 0.2"
fi

# Run it!
eval $CMD