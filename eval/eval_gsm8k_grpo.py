import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


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


def normalize_answer(ans):
    if ans is None:
        return None

    try:
        x = float(ans)
        if x.is_integer():
            return str(int(x))
        return str(x)
    except Exception:
        return str(ans).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--num_eval_samples", type=int, default=300)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--print_examples", type=int, default=0)
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
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    dataset = load_dataset("gsm8k", "main")["test"]
    dataset = dataset.select(range(min(args.num_eval_samples, len(dataset))))

    correct = 0
    total = 0
    total_tokens = 0
    printed = 0

    for ex in tqdm(dataset):
        question = ex["question"]
        gold = normalize_answer(extract_final_answer(ex["answer"]))

        messages = [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step. "
                    "End with either 'Final Answer: <number>' or '#### <number>'.\n\n"
                    f"{question}"
                ),
            }
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

        pred = normalize_answer(extract_final_answer(text))

        is_correct = pred == gold
        if is_correct:
            correct += 1

        total += 1
        total_tokens += len(generated)

        if printed < args.print_examples:
            print("\n" + "=" * 80)
            print("QUESTION:")
            print(question)
            print("\nMODEL OUTPUT:")
            print(text)
            print(f"\nPRED: {pred}")
            print(f"GOLD: {gold}")
            print(f"CORRECT: {is_correct}")
            print("=" * 80)
            printed += 1

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