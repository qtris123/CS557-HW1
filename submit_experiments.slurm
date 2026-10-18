#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J cs557_hw1
#SBATCH -o /home/vo43/HW1/slurm_%j.out
#SBATCH -e /home/vo43/HW1/slurm_%j.err

set -e

echo "=== Job started at $(date) on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load and activate conda environment
source /apps/external/conda/2025.09/etc/profile.d/conda.sh
conda activate cs557rl

cd /home/vo43/HW1

# Set SDL dummy driver for headless environment
export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1

echo "=== Running solution verification checks ==="
python verify_solution.py

echo "=== Running 12-experiment plan ==="
python run_experiments.py --execute

echo "=== Exporting plots and summary CSV ==="
python export_plots.py

echo "=== Job finished successfully at $(date) ==="
