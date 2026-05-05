import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def extract_final_answer(text):
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")

    m = re.findall(r"Final Answer:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].replace(",", "")

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--num_eval_samples", type=int, default=300)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    model_source = args.base_model if args.base_model else args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    dataset = load_dataset("gsm8k", "main")["test"]
    dataset = dataset.select(range(min(args.num_eval_samples, len(dataset))))

    correct = 0
    total = 0
    total_tokens = 0

    for ex in tqdm(dataset):
        question = ex["question"]
        gold = extract_final_answer(ex["answer"])

        messages = [
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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        pred = extract_final_answer(text)

        if pred == gold:
            correct += 1

        total += 1
        total_tokens += len(generated)

    acc = correct / total
    avg_tokens = total_tokens / total

    print("================================")
    print(f"Model: {args.model_path}")
    print(f"Eval samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Average generated tokens: {avg_tokens:.2f}")
    print("================================")


if __name__ == "__main__":
    main()
