#Only difference from regular train_grpo is utilization of two GPUs.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

    p.add_argument("--base_model", default="Qwen/Qwen3-4B")
    p.add_argument("--sft_adapter", default=None)
    p.add_argument(
        "--dataset_file",
        default="/storage/ice1/4/7/araj72/agentDistillation/gsm8k_qwen72b_gold_trajectories_h200.jsonl",
    )
    p.add_argument(
        "--output_dir",
        default="/storage/ice1/4/7/araj72/agentDistillation/qwen3_4b_sft_grpo_strong_final",
    )

    p.add_argument("--num_train_samples", type=int, default=6000)
    p.add_argument("--num_generations", type=int, default=4)

    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=192)

    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=3e-6)

    return p.parse_args()


def extract_final_answer(text):
    text = text.replace(",", "")

    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m[-1]

    m = re.findall(r"Final Answer:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return m[-1]

    m = re.findall(r"The answer is:?\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return m[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
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

    print("=" * 80)
    print("GRPO LARGE MODEL TRAINING")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Output: {args.output_dir}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max completion length: {args.max_completion_length}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base/SFT model...")
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
                "content": (
                    "You are a math reasoning agent. "
                    "Solve carefully but concisely. "
                    "End EXACTLY with: Final Answer: <number>. "
                    "Do not continue after the final answer."
                ),
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

    original_columns = dataset.column_names

    dataset = dataset.map(
        build_grpo_example,
        remove_columns=original_columns,
    )

    dataset = dataset.filter(lambda x: x["answer"] is not None)

    print(f"Using {len(dataset)} GRPO examples")
    print("Example prompt:")
    print(dataset[0]["prompt"][:500])
    print("Gold answer:", dataset[0]["answer"])

    def reward_func(completions, answer, **kwargs):
        rewards = []

        for completion, gt in zip(completions, answer):
            pred = extract_final_answer(completion)

            # Correctness is the main objective.
            acc_reward = 3.0 if pred == gt else 0.0

            # Encourage consistent final-answer formatting.
            format_reward = 0.2 if ("Final Answer" in completion or "####" in completion) else -0.2

            # Mild length penalty. Do not over-punish 4B for reasoning.
            length_penalty = 0.00025 * len(completion.split())

            reward = acc_reward + format_reward - length_penalty
            rewards.append(reward)

        return rewards

    config_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=999999,
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
            "temperature": 0.9,
            "top_p": 0.95,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
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