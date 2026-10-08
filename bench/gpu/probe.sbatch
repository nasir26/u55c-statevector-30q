#!/bin/bash
#SBATCH --job-name=jsc_gpu_probe
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=<WS>/logs/slurm/probe_%j.out
hostname; date
echo "--- workspace visible?"; ls -ld <WS> && ls <WS> | head -3
echo "--- GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv
nvidia-smi -q | grep -iE 'CUDA Version|Persistence|Compute Mode' | head
echo "--- CPU"; lscpu | grep -E 'Model name|Socket|Core\(s\)'; free -g | head -2
echo "--- tools"; which nvcc python3 conda 2>&1; ls /usr/local | grep -i cuda; python3 --version
