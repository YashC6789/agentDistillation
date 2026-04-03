#!/usr/bin/env python3
"""
Fine-tune a small Qwen 3 model on teacher trajectories with LoRA,
then evaluate it on GSM8K.
"""

import os
import re
import json
import random
import argparse
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--train_test_split", type=float, default=0.02)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--no_use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--gsm8k_split", type=str, default="test")
    parser.add_argument("--gsm8k_eval_limit", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a careful math reasoner. Show the reasoning briefly and end with 'Final answer: <number>'.",
    )

    args = parser.parse_args()

    if args.use_4bit and args.no_use_4bit:
        raise ValueError("Choose only one of --use_4bit or --no_use_4bit.")

    if not args.use_4bit and not args.no_use_4bit:
        args.use_4bit = True

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no}: {exc}") from exc
    return rows


def normalize_record(rec: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    if "messages" in rec and isinstance(rec["messages"], list):
        return {"messages": rec["messages"]}

    if "prompt" in rec and "completion" in rec:
        return {
            "messages": [
                {"role": "user", "content": str(rec["prompt"])},
                {"role": "assistant", "content": str(rec["completion"])},
            ]
        }

    if "instruction" in rec and "output" in rec:
        user_msg = str(rec["instruction"])
        if rec.get("input"):
            user_msg += "\n\nAdditional input:\n" + str(rec["input"])
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": str(rec["output"])},
            ]
        }

    if "input" in rec and "output" in rec:
        return {
            "messages": [
                {"role": "user", "content": str(rec["input"])},
                {"role": "assistant", "content": str(rec["output"])},
            ]
        }

    raise ValueError(f"Unsupported record schema keys: {list(rec.keys())}")


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned = []
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = m.get("content", "")
        if role not in {"system", "user", "assistant"}:
            continue
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        content = str(content).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def prepare_sft_dataset(records: List[Dict[str, Any]], tokenizer) -> Dataset:
    rows = []
    dropped = 0

    for rec in records:
        norm = normalize_record(rec)
        messages = clean_messages(norm["messages"])
        roles = [m["role"] for m in messages]

        if "user" not in roles or "assistant" not in roles:
            dropped += 1
            continue

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        rows.append({"text": text})

    print(f"Prepared {len(rows)} training examples; dropped {dropped}.")
    return Dataset.from_list(rows)


def last_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def gsm8k_gold_answer(ans_text: str) -> Optional[str]:
    if "####" in ans_text:
        tail = ans_text.split("####")[-1].strip()
        return last_number(tail)
    return last_number(ans_text)


def build_eval_prompt(system_prompt: str, question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def infer_dtype(args: argparse.Namespace):
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = infer_dtype(args)

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype in (torch.bfloat16, torch.float16) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    return model, tokenizer, dtype


def generate_one(model, tokenizer, messages, max_new_tokens: int = 256) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")

    print("\nLoading model and tokenizer...")
    model, tokenizer, dtype = load_model_and_tokenizer(args)
    print(f"Model loaded: {args.model_name}")
    print(f"Using dtype: {dtype}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nReading teacher trajectories...")
    teacher_records = read_jsonl(args.teacher_path)
    print(f"Raw teacher records: {len(teacher_records)}")
    if teacher_records:
        print(f"Example keys: {list(teacher_records[0].keys())}")

    teacher_dataset = prepare_sft_dataset(teacher_records, tokenizer)
    teacher_dataset = teacher_dataset.shuffle(seed=args.seed)

    split = teacher_dataset.train_test_split(
        test_size=args.train_test_split,
        seed=args.seed,
    )
    train_ds = split["train"]
    valid_ds = split["test"]

    print(f"Train size: {len(train_ds)}")
    print(f"Valid size: {len(valid_ds)}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    train_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        #max_seq_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting training...")
    train_result = trainer.train()
    print(train_result)

    print("\nSaving adapter and tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nLoading GSM8K...")
    gsm8k = load_dataset("openai/gsm8k", "main", split=args.gsm8k_split)
    if args.gsm8k_eval_limit is not None and args.gsm8k_eval_limit > 0:
        gsm8k = gsm8k.select(range(min(args.gsm8k_eval_limit, len(gsm8k))))

    print(f"GSM8K size used: {len(gsm8k)}")

    print("\nEvaluating on GSM8K...")
    results = []
    correct = 0

    for idx, ex in enumerate(gsm8k):
        question = ex["question"]
        gold = gsm8k_gold_answer(ex["answer"])

        pred_text = generate_one(
            trainer.model,
            tokenizer,
            build_eval_prompt(args.system_prompt, question),
            max_new_tokens=args.max_new_tokens,
        )
        pred = last_number(pred_text)
        is_correct = (pred == gold)
        correct += int(is_correct)

        results.append(
            {
                "idx": idx,
                "question": question,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_output": pred_text,
            }
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(gsm8k):
            acc = correct / (idx + 1)
            print(f"{idx + 1}/{len(gsm8k)} done | accuracy={acc:.4f}")

    final_acc = correct / len(results) if results else 0.0
    print(f"\nFinal GSM8K exact-match accuracy: {final_acc:.4f}")

    results_path = os.path.join(args.output_dir, "gsm8k_eval_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_name": args.model_name,
        "teacher_path": args.teacher_path,
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
        "gsm8k_split": args.gsm8k_split,
        "gsm8k_eval_limit": args.gsm8k_eval_limit,
        "accuracy": final_acc,
        "num_eval_examples": len(results),
        "use_4bit": args.use_4bit,
        "dtype": str(dtype),
    }
    save_json(os.path.join(args.output_dir, "run_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
