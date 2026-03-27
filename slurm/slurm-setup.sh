#!/bin/bash
#SBATCH --job-name=trinity-setup
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=slurm/logs/setup_%j.out
#SBATCH --error=slurm/logs/setup_%j.err

# Create log directory (may not exist on first run)
mkdir -p slurm/logs

srun bash slurm/setup-env.sh
