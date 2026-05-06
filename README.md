# agentDistillation

Distilling reasoning from a large teacher model into smaller models using SFT + GRPO on GSM8K.

## Setup

Recommended:
- Python 3.11 (conda env)

### Trajectory generation

pip install vllm datasets autoawq


### SFT

pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

Optional (if autoawq causes issues):

pip uninstall autoawq -y


### GRPO

pip install -U trl transformers accelerate datasets peft


## Data

- `gsm8k_golden_trajectories.jsonl` → small (~300 samples, testing)
- `gsm8k_qwen72b_gold_trajectories_h200.jsonl` → full dataset (~7k samples, best results)

Teacher model used for trajectories:
- Qwen2.5-72B-Instruct-AWQ

## Training

### SFT

Test run:

python train_sft.py
--model_name Qwen/Qwen3-0.6B
--dataset_file gsm8k_golden_trajectories.jsonl


Full run:

python train_sft.py
--model_name Qwen/Qwen3-0.6B
--dataset_file /path/to/gsm8k_qwen72b_gold_trajectories_h200.jsonl
--output_dir /path/to/output
--merged_save_path /path/to/model
--num_train_samples 6000
--max_steps 500


### GRPO


python train_grpo.py
--base_model /path/to/sft_model
--dataset_file /path/to/gsm8k_qwen72b_gold_trajectories_h200.jsonl
--num_train_samples 5000
--num_generations 4
--max_completion_length 160
--max_steps 800
--output_dir /path/to/grpo_model


## Evaluation


python eval_gsm8k_grpo.py
--model_path /path/to/model
--num_eval_samples 300
--max_new_tokens 160


## Results

| Model | Method | Accuracy | Avg Tokens |
|------|--------|----------|------------|
| Qwen3-0.6B | SFT | 35.33% | 145.27 |
| Qwen3-0.6B | GRPO | **53.00%** | **110.56** |
| Qwen3-1.7B | SFT | 40.00% | 146.40 |
| Qwen3-1.7B | GRPO | **58.00%** | **132.65** |
| Qwen3-4B | SFT | 44.33% | 146.76 |
| Qwen3-4B | GRPO (strong) | **67.00%** | 154.34 |

## Notes

- Update file paths in scripts to match your environment
- Do not commit model checkpoints (very large)
- Add `.gitignore` for:
  - model weights
  - HF cache
  - training outputs

## Summary

- GRPO significantly improves accuracy, especially for smaller models
- 1.7B is a strong balance of performance and efficiency
- 4B requires stronger GRPO settings to fully benefit