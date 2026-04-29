#!/bin/bash
# Launch a vol-scaled reward sweep on the SLURM cluster.
# Tests vol_scaled vs sortino reward with varying inaction_penalty.
#
# Usage:
#   bash slurm/launch-vol-scaled-sweep.sh        # 4 parallel agents
#   bash slurm/launch-vol-scaled-sweep.sh 2      # override: 2 agents

cd ~/trinity
export PATH="$PWD/env/bin:$PATH"

set -e

# Load W&B credentials
if [[ -f slurm/.env ]]; then
    set -a; source slurm/.env; set +a
else
    echo "ERROR: slurm/.env not found. Copy slurm/.env.template and fill in credentials."
    exit 1
fi

N_AGENTS=${1:-4}

echo "=== Creating Vol-Scaled Reward W&B Sweep ==="
set +e
SWEEP_OUTPUT=$(env/bin/python -m wandb sweep configs/vol_scaled_sweep.yaml 2>&1)
SWEEP_EXIT=$?
set -e
echo "$SWEEP_OUTPUT"

if [[ $SWEEP_EXIT -ne 0 ]]; then
    echo "ERROR: wandb sweep failed (exit code $SWEEP_EXIT)."
    exit 1
fi

# Extract sweep ID (format: wandb agent entity/project/sweep_id)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '[\w-]+/[\w-]+/[\w]+$' | tail -1)

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: Could not extract sweep ID from wandb output."
    echo "Try running 'env/bin/python -m wandb sweep configs/vol_scaled_sweep.yaml' manually."
    echo "Then launch agents with: sbatch --array=1-${N_AGENTS} --export=SWEEP_ID=<id> slurm/slurm-sweep-ppo.sh"
    exit 1
fi

echo ""
echo "Sweep ID: $SWEEP_ID"
echo "Launching ${N_AGENTS} parallel GPU agents..."
echo ""

JOB_ID=$(sbatch --array=1-${N_AGENTS} --export=SWEEP_ID="${SWEEP_ID}" slurm/slurm-sweep-ppo.sh | awk '{print $NF}')

echo "=== Vol-Scaled Reward Sweep Launched ==="
echo "SLURM Job Array ID: ${JOB_ID}"
echo "Agents: ${N_AGENTS}"
echo "W&B Sweep ID: ${SWEEP_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER                    # SLURM job status"
echo "  wandb sweep ${SWEEP_ID}             # W&B sweep dashboard"
echo "  cat slurm/logs/ppo_sweep_*          # job logs"
