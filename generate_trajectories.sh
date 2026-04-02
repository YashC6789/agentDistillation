#!/bin/bash
#SBATCH --job-name=generate_T          # Job name
#SBATCH --output=genTraj%j.out                # Standard output file
#SBATCH --error=genTraj%j.err                 # Error file
#SBATCH --partition=ice-gpu              # Partition name
#SBATCH -N1 --gres=gpu:4                 # Request 1 GPU (not used, but kept as requested)
#SBATCH --cpus-per-task=4                # Request 4 CPU cores
#SBATCH --mem-per-gpu=128GB              # Request 16GB RAM
#SBATCH --time=05:00:00                  # Max job runtime
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ychauhan9@gatech.edu

set -euo pipefail

module load python/3.11
module load anaconda3
module load uv || true

# ===== SCRATCH REDIRECT (DO NOT REMOVE) =====
export SCRATCH_ROOT="/home/hice1/ychauhan9/scratch/8803Project"

export HF_HOME="$SCRATCH_ROOT/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export XINFERENCE_HOME="$SCRATCH_ROOT/xinference"
export VLLM_CACHE_ROOT="$SCRATCH_ROOT/vllm"

export TORCH_HOME="$SCRATCH_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$SCRATCH_ROOT/torch_extensions"

export XDG_CACHE_HOME="$SCRATCH_ROOT/.cache"
export PYTHONPYCACHEPREFIX="$SCRATCH_ROOT/pycache"
export PIP_CACHE_DIR="$SCRATCH_ROOT/pip"

export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p \
  "$SCRATCH_ROOT"/{hf,xinference,vllm,torch,torch_extensions,.cache,pip,runs}

# ===== Activate environment =====
cd "$SCRATCH_ROOT"
source agent_env/bin/activate

# ===== Generate Trajectories =====

python generate_hf_trajectories.py

echo "Trajectories Generated"