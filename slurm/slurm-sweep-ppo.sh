#!/bin/bash
#SBATCH --job-name=ppo-sweep
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/ppo_sweep_%A_%a.out
#SBATCH --error=slurm/logs/ppo_sweep_%A_%a.err

# Single PPO sweep agent. Launched as part of a SLURM job array.
# Each array task is an independent W&B agent that pulls hyperparameter
# configurations from the sweep queue and trains one trial at a time.
#
# SWEEP_ID is passed via --export when sbatch is called from launch-ppo-sweep.sh.

mkdir -p slurm/logs

# Source shared environment setup
source slurm/common.sh

echo "=== PPO Sweep Agent (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Sweep ID: ${SWEEP_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: SWEEP_ID not set. Launch via slurm/launch-ppo-sweep.sh"
    exit 1
fi

# Run the W&B sweep agent — it will pull configs and run sweep_train.py
wandb agent "$SWEEP_ID"
