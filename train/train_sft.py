import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--dataset_file",
        default="/storage/ice1/4/7/araj72/agentDistillation/gsm8k_qwen72b_gold_trajectories_h200.jsonl",
    )
    p.add_argument(
        "--output_dir",
        default="/storage/ice1/4/7/araj72/agentDistillation/outputs_sft_qwen3_0_6b_h200",
    )
    p.add_argument(
        "--merged_save_path",
        default="/storage/ice1/4/7/araj72/agentDistillation/qwen3_0_6b_sft_qwen72b_gold",
    )

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--num_train_samples", type=int, default=6000)

    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=2e-4)

    return p.parse_args()


def main():
    args = parse_args()

    os.environ["WANDB_DISABLED"] = "true"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.merged_save_path, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    print("=" * 70)
    print("Plain HF/TRL LoRA SFT")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Merged save path: {args.merged_save_path}")
    print(f"BF16: {use_bf16}, FP16: {use_fp16}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.dataset_file, split="train")

    if args.num_train_samples > 0:
        dataset = dataset.select(range(min(args.num_train_samples, len(dataset))))

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=False,
            )
            for m in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    print(f"Training samples: {len(dataset)}")
    print("Example text:")
    print(dataset[0]["text"][:700])

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        seed=42,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        # New SFT args
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("=" * 70)
    print("STARTING SFT")
    print("=" * 70)

    trainer.train()

    print("=" * 70)
    print("SFT COMPLETE")
    print("=" * 70)

    print("Saving LoRA adapter...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Merging LoRA into base model...")
    merged_model = trainer.model.merge_and_unload()

    print("Saving merged model...")
    merged_model.save_pretrained(args.merged_save_path, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_save_path)

    print(f"Done. Adapter saved to: {args.output_dir}")
    print(f"Done. Merged model saved to: {args.merged_save_path}")


if __name__ == "__main__":
    main()