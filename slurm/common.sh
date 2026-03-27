#!/bin/bash
# Shared setup sourced by all SLURM run scripts.
# Activates the virtual environment and loads W&B credentials.

set -e

# Resolve project root (one level up from slurm/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "${PROJECT_ROOT}/env/bin/activate"

# Load W&B credentials from .env
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "WARNING: ${ENV_FILE} not found. W&B logging may fail."
    echo "Copy slurm/.env.template to slurm/.env and fill in your credentials."
fi

# Ensure we're in the project root (so relative paths in configs work)
cd "$PROJECT_ROOT"

echo "=== Environment ==="
echo "Python:       $(which python3)"
echo "Project root: $PROJECT_ROOT"
echo "WANDB_ENTITY: ${WANDB_ENTITY:-unset}"
echo "WANDB_PROJECT: ${WANDB_PROJECT:-unset}"
echo "Device:       $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "=================="
