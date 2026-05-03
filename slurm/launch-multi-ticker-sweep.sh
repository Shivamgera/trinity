#!/bin/bash
# Launch a multi-ticker training sweep on the SLURM cluster.
# Trains on 5 tickers (AAPL, MSFT, GOOGL, SPY, AMZN) round-robin.
# Val/test evaluation remains AAPL-only.
#
# IMPORTANT: Run scripts/download_data.py --multi-ticker on the cluster
# first to ensure all ticker data files exist.
#
# Usage:
#   bash slurm/launch-multi-ticker-sweep.sh        # 4 parallel agents
#   bash slurm/launch-multi-ticker-sweep.sh 2      # override: 2 agents

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

# Verify multi-ticker data exists
for TICKER in msft googl spy amzn; do
    if [[ ! -f "data/processed/${TICKER}_features.parquet" ]]; then
        echo "ERROR: Missing data/processed/${TICKER}_features.parquet"
        echo "Run: env/bin/python -m scripts.download_data --multi-ticker"
        exit 1
    fi
done
echo "All ticker data files verified."

N_AGENTS=${1:-4}

echo "=== Creating Multi-Ticker Training W&B Sweep ==="
set +e
SWEEP_OUTPUT=$(env/bin/python -m wandb sweep configs/multi_ticker_sweep.yaml 2>&1)
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
    echo "Try running 'env/bin/python -m wandb sweep configs/multi_ticker_sweep.yaml' manually."
    echo "Then launch agents with: sbatch --array=1-${N_AGENTS} --export=SWEEP_ID=<id> slurm/slurm-sweep-ppo.sh"
    exit 1
fi

echo ""
echo "Sweep ID: $SWEEP_ID"
echo "Launching ${N_AGENTS} parallel GPU agents..."
echo ""

JOB_ID=$(sbatch --array=1-${N_AGENTS} --export=SWEEP_ID="${SWEEP_ID}" slurm/slurm-sweep-ppo.sh | awk '{print $NF}')

echo "=== Multi-Ticker Training Sweep Launched ==="
echo "SLURM Job Array ID: ${JOB_ID}"
echo "Agents: ${N_AGENTS}"
echo "W&B Sweep ID: ${SWEEP_ID}"
echo "Training tickers: AAPL, MSFT, GOOGL, SPY, AMZN"
echo "Eval ticker: AAPL only"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER                    # SLURM job status"
echo "  wandb sweep ${SWEEP_ID}             # W&B sweep dashboard"
echo "  cat slurm/logs/ppo_sweep_*          # job logs"
