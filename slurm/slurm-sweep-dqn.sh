#!/bin/bash
#SBATCH --job-name=dqn-sweep
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/dqn_sweep_%A_%a.out
#SBATCH --error=slurm/logs/dqn_sweep_%A_%a.err

# Single DQN sweep agent. Launched as part of a SLURM job array.
# Each array task is an independent W&B agent that pulls hyperparameter
# configurations from the sweep queue and trains one trial at a time.
#
# SWEEP_ID is passed via --export when sbatch is called from launch-dqn-sweep.sh.

cd ~/trinity
export PATH="$PWD/env/bin:$PATH"

mkdir -p slurm/logs

# Load W&B credentials
if [[ -f slurm/.env ]]; then
    set -a; source slurm/.env; set +a
fi

echo "=== DQN Sweep Agent (array task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Sweep ID: ${SWEEP_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: SWEEP_ID not set. Launch via slurm/launch-dqn-sweep.sh"
    exit 1
fi

# Run the W&B sweep agent — it will pull configs and run sweep_train_dqn.py
env/bin/python -m wandb agent "$SWEEP_ID"
