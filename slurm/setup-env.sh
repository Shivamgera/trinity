#!/bin/bash
# Setup Python virtual environment for the SLURM cluster.
# Uses --system-site-packages to inherit the cluster's PyTorch/CUDA
# installation, avoiding a 2+ GB re-download.

set -e

echo "=== Setting up Python virtual environment ==="
python3 -m venv --system-site-packages env
source env/bin/activate

echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Install the project in editable mode (pulls all deps from pyproject.toml)
pip install --upgrade pip
pip install -e .

# Verify key packages
python3 -c "import stable_baselines3; print(f'stable-baselines3 {stable_baselines3.__version__}')"
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import wandb; print(f'wandb {wandb.__version__}')"

echo "=== Environment setup complete ==="
