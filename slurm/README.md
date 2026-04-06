# SLURM GPU Cluster Setup

Scripts for running RL hyperparameter sweeps and multi-seed training on the DSV Olympus GPU cluster.

## Quick Start

All commands below are executed **on the Olympus cluster**.

### 1. Clone and enter the repository

```bash
git clone <your-repo-url>
cd research1
```

### 2. Set up the Python environment

```bash
sbatch slurm/slurm-setup.sh
```

Wait for completion: `squeue -u $USER` (should disappear when done).
Check the log: `cat slurm/logs/setup_*.out`

### 3. Validate GPU access

```bash
sbatch slurm/validate-gpu.sh
```

Check output: `cat slurm/logs/gpu_check_*.out` — should show `CUDA available: True`.

### 4. Configure W&B credentials

```bash
cp slurm/.env.template slurm/.env
nano slurm/.env   # fill in WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
```

Get your API key from: https://wandb.ai/authorize

### 5. Ensure data files are present

The training scripts expect both raw and processed data:
```
data/raw/aapl_ohlcv.parquet
data/processed/aapl_features.parquet
data/processed/headlines.json
data/processed/precomputed_signals.json
```

If these files are gitignored, copy them to the cluster:
```bash
scp -r data/raw/ olympus:~/trinity/data/raw/
scp -r data/processed/ olympus:~/trinity/data/processed/
```

## Running Sweeps

### DQN Hyperparameter Sweep (50 trials, 8 parallel agents)

```bash
bash slurm/launch-dqn-sweep.sh       # creates sweep + launches 8 GPU jobs
bash slurm/launch-dqn-sweep.sh 4     # override: 4 parallel agents
```

### PPO Hyperparameter Sweep (20 trials, 8 parallel agents)

```bash
bash slurm/launch-ppo-sweep.sh       # creates sweep + launches 8 GPU jobs
bash slurm/launch-ppo-sweep.sh 4     # override: 4 parallel agents
```

### Manual sweep launch (if auto-extraction fails)

```bash
# Create the sweep manually
wandb sweep configs/dqn_sweep.yaml
# Note the sweep ID from the output (e.g. your-entity/robust-trinity/abc123)

# Launch agents
sbatch --array=1-8 --export=SWEEP_ID="your-entity/robust-trinity/abc123" slurm/slurm-sweep-dqn.sh
```

## Running Multi-Seed Training

After sweeps complete and you've locked the best hyperparameters:

### DQN (20 seeds in parallel)

```bash
sbatch slurm/slurm-multiseed-dqn.sh              # all 20 seeds
sbatch --array=0-3 slurm/slurm-multiseed-dqn.sh  # first 4 seeds only
```

### PPO (20 seeds in parallel)

```bash
sbatch slurm/slurm-multiseed-ppo.sh              # all 20 seeds
sbatch --array=0-3 slurm/slurm-multiseed-ppo.sh  # first 4 seeds only
```

### After multi-seed jobs complete

Evaluate and freeze top seeds (run locally or on cluster):
```bash
python3 -m scripts.run_multiseed_dqn --eval-only --freeze
python3 -m scripts.run_multiseed --eval-only --freeze
```

## Monitoring

```bash
squeue -u $USER                          # list your running/queued jobs
scontrol show job <JOB_ID>               # detailed job info
scancel <JOB_ID>                         # cancel a job
scancel --user=$USER                     # cancel all your jobs
cat slurm/logs/dqn_sweep_<ARRAY>_<TASK>.out  # view job output
sinfo -p gpu                             # GPU partition status
```

W&B dashboard: https://wandb.ai — sweeps appear under your project automatically.

## Resource Allocation per Job

| Resource | Value |
|----------|-------|
| GPUs | 1 |
| CPUs | 4 |
| Memory | 8 GB |
| Time limit | 4 hours |
| Partition | gpu |
| Account | slurm-students |

## File Overview

| File | Type | Purpose |
|------|------|---------|
| `setup-env.sh` | Shell | Create venv + install dependencies |
| `slurm-setup.sh` | SBATCH | Run setup as SLURM job |
| `validate-gpu.sh` | SBATCH | Verify GPU/CUDA access |
| `.env.template` | Config | W&B credentials template |
| `common.sh` | Shell | Shared: venv activation + W&B auth |
| `launch-dqn-sweep.sh` | Shell | Create DQN sweep + launch agents |
| `slurm-sweep-dqn.sh` | SBATCH | Single DQN sweep agent |
| `launch-ppo-sweep.sh` | Shell | Create PPO sweep + launch agents |
| `slurm-sweep-ppo.sh` | SBATCH | Single PPO sweep agent |
| `slurm-multiseed-dqn.sh` | SBATCH | 20-seed DQN array job |
| `slurm-multiseed-ppo.sh` | SBATCH | 20-seed PPO array job |

## Troubleshooting

**"CUDA is NOT available"** — The GPU partition may not have allocated a GPU. Check `scontrol show job <ID>` for `Gres=gpu:1`.

**W&B login fails** — Ensure `slurm/.env` has a valid `WANDB_API_KEY`. Test with `wandb login --verify` in an interactive session.

**Job killed (OOM)** — Increase `--mem` in the SBATCH script (e.g., `--mem=16G`).

**Job killed (time limit)** — Increase `--time` (e.g., `--time=08:00:00`). Early stopping should fire before 4h for these model sizes.

**"Module not found"** — Ensure `sbatch slurm/slurm-setup.sh` completed successfully. Check `cat slurm/logs/setup_*.out`.

**Offline mode** — If the cluster lacks internet, set `WANDB_MODE=offline` in `slurm/.env` and sync logs later with `wandb sync wandb/offline-run-*`.
