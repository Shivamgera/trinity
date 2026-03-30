#!/bin/bash
#SBATCH --job-name=dqn-multiseed
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm/logs/dqn_ms_%A_%a.out
#SBATCH --error=slurm/logs/dqn_ms_%A_%a.err

# Multi-seed DQN training via SLURM array.
# Each array task trains ONE seed out of 20, all in parallel on separate GPUs.
# Seeds match the ALL_SEEDS list in scripts/run_multiseed_dqn.py:
#   [42, 123, 456, 789, 999, 1024, 2048, 3141, 4096, 5555,
#    7777, 8888, 9999, 1111, 2222, 3333, 4444, 5678, 6789, 7890]
#
# Usage:
#   sbatch slurm/slurm-multiseed-dqn.sh              # train all 20 seeds
#   sbatch --array=0-3 slurm/slurm-multiseed-dqn.sh  # train only first 4 seeds
#
# After all jobs complete, evaluate + freeze top seeds locally:
#   python3 -m scripts.run_multiseed_dqn --eval-only --freeze

mkdir -p slurm/logs

# Source shared environment setup
source slurm/common.sh

# Map array task ID to seed
SEEDS=(42 123 456 789 999 1024 2048 3141 4096 5555 7777 8888 9999 1111 2222 3333 4444 5678 6789 7890)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== DQN Multi-Seed Training ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${SEED}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Train the single seed using a small Python wrapper that calls train_seed()
python3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
from scripts.run_multiseed_dqn import train_seed, _get_normalizer
normalizer = _get_normalizer()
result = train_seed(seed=${SEED}, normalizer=normalizer, force=False)
print(f'Result: {result}')
"

echo "=== DQN Seed ${SEED} Complete ==="
