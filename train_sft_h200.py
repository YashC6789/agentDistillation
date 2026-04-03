import os
import torch

# Keep single GPU for stability; 0.5B does not need multi-GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import transformers
import transformers.activations
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Patch for transformers compatibility
if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

# ---------------- CONFIG ----------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
max_seq_length = 2048

dataset_file = os.path.expanduser(
    "~/scratch/araj72/agentDistillation/gsm8k_platinum_trajectories.jsonl"
)

output_dir = os.path.expanduser(
    "~/scratch/araj72/agentDistillation/outputs_sft_h200"
)

save_path = os.path.expanduser(
    "~/scratch/araj72/agentDistillation/qwen_0.5b_sft_midway_h200"
)

# ---------------- LOAD MODEL ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.float16,
)

# ---------------- PEFT / LoRA ----------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
)

# ---------------- LOAD DATASET ----------------
dataset = load_dataset("json", data_files=dataset_file, split="train")

def formatting_prompts_func(examples):
    instructions = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=False
        )
        for m in instructions
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"Loaded {len(dataset)} training examples")

# ---------------- TRAINER ----------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=8,      # H200 can handle this for 0.5B 4-bit
        gradient_accumulation_steps=2,      # effective batch size 16
        warmup_steps=5,
        max_steps=60,                       # lower to 30 if short on time
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        output_dir=output_dir,
        report_to="none",
    ),
)

# ---------------- TRAIN ----------------
print("Starting SFT training on H200...")
trainer.train()

# ---------------- SAVE ----------------
# Fastest save option: LoRA adapters only
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Success! LoRA adapter + tokenizer saved to: {save_path}")

# If you specifically need a merged model, use this instead,
# but it will take longer and use more disk:
# model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")