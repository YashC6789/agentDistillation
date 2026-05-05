import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
import inspect


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--sft_adapter", default=None)
    p.add_argument("--dataset_file", default=os.path.expanduser("~/scratch/araj72/agentDistillation/gsm8k_qwen72b_gold_trajectories_h200.jsonl"))
    p.add_argument("--output_dir", default=os.path.expanduser("~/scratch/araj72/agentDistillation/grpo_qwen3_0_6b"))
    p.add_argument("--num_train_samples", type=int, default=50)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=30)
    return p.parse_args()


def extract_final_answer(text):
    # Prefer GSM8K-style #### answer
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")

    # Prefer Final Answer:
    m = re.findall(r"Final Answer:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].replace(",", "")

    # Fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def get_question_and_answer(messages):
    question = ""
    gold_answer = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            question = content
        elif role == "assistant":
            gold_answer = content

    return question, gold_answer


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.sft_adapter:
        print(f"Loading SFT adapter from {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)

    print("Loading trajectory dataset...")
    dataset = load_dataset("json", data_files=args.dataset_file, split="train")

    if args.num_train_samples > 0:
        dataset = dataset.select(range(min(args.num_train_samples, len(dataset))))

    def build_grpo_example(example):
        messages = example["messages"]
        question, gold_answer = get_question_and_answer(messages)

        prompt_messages = [
            {
                "role": "system",
                "content": "You are a math reasoning agent. Solve briefly. End EXACTLY with: Final Answer: <number>. Do not continue after that.",
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt,
            "answer": extract_final_answer(gold_answer),
            "gold_text": gold_answer,
        }

    dataset = dataset.map(build_grpo_example)
    dataset = dataset.filter(lambda x: x["answer"] is not None)

    print(f"Using {len(dataset)} GRPO examples")
    print("Example prompt:")
    print(dataset[0]["prompt"][:500])
    print("Gold answer:", dataset[0]["answer"])

    def reward_func(completions, answer, **kwargs):
        rewards = []

        for completion, gt in zip(completions, answer):
            pred = extract_final_answer(completion)

            acc_reward = 1.0 if pred == gt else 0.0
            format_reward = 0.1 if "Final Answer" in completion else -0.1
            length_penalty = 0.001 * len(completion.split())

            rewards.append(acc_reward + format_reward - length_penalty)

        return rewards

    config_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=25,
        bf16=True,
        report_to="none",
    )

    sig = inspect.signature(GRPOConfig).parameters

    if "max_prompt_length" in sig:
        config_kwargs["max_prompt_length"] = args.max_prompt_length

    if "max_completion_length" in sig:
        config_kwargs["max_completion_length"] = args.max_completion_length

    if "generation_kwargs" in sig:
        config_kwargs["generation_kwargs"] = {
            "max_new_tokens": args.max_completion_length,
            "do_sample": True,
            "temperature": 0.7,
            "eos_token_id": tokenizer.eos_token_id,
        }

    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Saving GRPO model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()