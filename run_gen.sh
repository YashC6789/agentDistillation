#!/bin/bash
#SBATCH --job-name=qwen_gen
#SBATCH --output=gen_%j.out
#SBATCH --error=gen_%j.err
#SBATCH --partition=ice-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -e

# Setup Scratch Environment
export SCRATCH_ROOT="/home/hice1/araj72/scratch/agentDistillation"
mkdir -p "$SCRATCH_ROOT/hf_cache"
export HF_HOME="$SCRATCH_ROOT/hf_cache"

# Activate your env
source /home/hice1/araj72/scratch/agentDistillation/agent_env/bin/activate

# Install requirements if not present
pip install vllm datasets autoawq

# Run the generation
python generate_hf_trajectories_temp.py