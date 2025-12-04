#!/bin/bash
#SBATCH --job-name=sam3d_demo
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --time=02:00:00          # Adjust based on your needs
#SBATCH --cpus-per-task=32 
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --partition=RTXA6Kq           # Change to your GPU partition name

# Load necessary modules (adjust based on your HPC setup)
module load cuda12.6/toolkit      # As mentioned in doc/uv_setup.md
module load cudnn9.8-cuda12.8/9.8.0.87
# module load python/3.11         # If needed on your system

nvidia-smi

# Navigate to project directory
cd /export/home2/chongtia002/sam-3d-objects

export TMPDIR=${PWD}/.tmp
export TEMP=${PWD}/.tmp

export XFORMERS_FORCE_DISABLE_TRITON==true
# Set up environment variables if needed
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export LD_LIBRARY_PATH=${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}

# Run the demo using uv
# uv run demo.py
uv run demo.py
