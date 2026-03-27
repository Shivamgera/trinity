#!/bin/bash
#SBATCH --job-name=trinity-gpu-check
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm/logs/gpu_check_%j.out
#SBATCH --error=slurm/logs/gpu_check_%j.err

mkdir -p slurm/logs
source env/bin/activate

echo "=== GPU Validation ==="
python3 -c "
import torch
print(f'torch version:       {torch.__version__}')
print(f'CUDA available:      {torch.cuda.is_available()}')
print(f'CUDA device count:   {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device:      {torch.cuda.current_device()}')
    print(f'Device name:         {torch.cuda.get_device_name(0)}')
    print(f'Memory allocated:    {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
    print(f'Memory reserved:     {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
    # Quick tensor test on GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'GPU matmul test:     OK (result shape {y.shape})')
else:
    print('ERROR: CUDA is NOT available!')
    exit(1)
"
echo "=== GPU Validation Complete ==="
