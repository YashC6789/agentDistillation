import os
# FORCE SINGLE GPU TO BYPASS UNSLOTH MULTI-GPU BUG ON V100
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import transformers.activations
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Patch for 2026 Transformers compatibility
if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

# 1. Configuration
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
max_seq_length = 2048
dataset_file = os.path.expanduser("~/scratch/araj72/agentDistillation/gsm8k_golden_trajectories_new.jsonl")

# 2. Load Model (1 GPU is plenty for 0.5B)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 3. Add PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = False, 
)

# 4. Load & Format Dataset
dataset = load_dataset("json", data_files=dataset_file, split="train")

def formatting_prompts_func(examples):
    instructions = examples["messages"]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in instructions]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Set up Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 4, 
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs_sft",
        save_strategy = "no",
    ),
)

# 6. Train!
print("Starting SFT Training (Single-GPU Bypass)...")
trainer.train()

# 7. Save merged model
save_path = os.path.expanduser("~/scratch/araj72/agentDistillation/qwen_0.5b_sft_midway")
model.save_pretrained_merged(save_path, tokenizer, save_method = "merged_16bit")
print(f"Success! Model saved to: {save_path}")