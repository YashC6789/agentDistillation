#!/bin/bash
#SBATCH --job-name=fine_tune          # Job name
#SBATCH --output=fTj%j.out                # Standard output file
#SBATCH --error=fT%j.err                 # Error file
#SBATCH --partition=ice-gpu              # Partition name
#SBATCH --gres=gpu:H100:1                 # Request 1 GPU (not used, but kept as requested)
#SBATCH --cpus-per-task=4                # Request 4 CPU cores
#SBATCH --mem-per-gpu=128GB              # Request 16GB RAM
#SBATCH --time=03:00:00                  # Max job runtime
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

SCRIPT_PATH="$SCRATCH_ROOT/train_qwen3_distill_gsm8k.py"
TEACHER_PATH="$SCRATCH_ROOT/trajectories/gsm8k_qwen72b_gold_trajectories_h200.jsonl"
OUTPUT_DIR="$SCRATCH_ROOT/outputs/qwen3_17b"

python "$SCRIPT_PATH" \
  --teacher_path "$TEACHER_PATH" \
  --model_name "Qwen/Qwen3-1.7B" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 2 \
  --per_device_batch_size 2 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --max_seq_len 1024 \
  --gsm8k_eval_limit 200 \
  --use_4bit

echo "Job finished on $(date)"
