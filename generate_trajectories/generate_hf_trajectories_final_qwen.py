import json
import re
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ---------------- CONFIGURATION ----------------
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"
NUM_EXAMPLES = 7473
NUM_GPUS = 2
BATCH_SIZE = 64
MAX_MODEL_LEN = 4096
MAX_TOKENS = 1024
TEMPERATURE = 0.3
TOP_P = 0.9
GPU_MEMORY_UTILIZATION = 0.90

PROJECT_ROOT = os.path.expanduser("~/scratch/araj72/agentDistillation")
HF_CACHE = os.path.join(PROJECT_ROOT, "hf_cache")
VLLM_CACHE = os.path.join(PROJECT_ROOT, "vllm_cache")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "gsm8k_qwen72b_gold_trajectories_h200.jsonl")

OVERWRITE_OUTPUT = True
ASSISTANT_PREFIX = ""


def extract_ground_truth(answer_str: str):
    parts = answer_str.split("####")
    if len(parts) == 2:
        return parts[1].strip().replace(",", "")
    return None


def extract_model_answer(text: str):
    text = text.replace(",", "")

    explicit = re.search(
        r"The answer is:\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1)

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    return None


def build_prompt(question: str):
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "Show your work briefly and end with exactly: The answer is: [number]\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let's think step by step.\n"
    )


def main():
    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs(VLLM_CACHE, exist_ok=True)
    os.makedirs(PROJECT_ROOT, exist_ok=True)

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["VLLM_CACHE_ROOT"] = VLLM_CACHE

    if OVERWRITE_OUTPUT and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print("\n--- H200 Qwen Teacher Pipeline ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Model:        {MODEL_ID}")
    print(f"GPUs:         {NUM_GPUS}x H200")
    print(f"Examples:     {NUM_EXAMPLES}")
    print(f"Batch Size:   {BATCH_SIZE}")
    print(f"Output File:  {OUTPUT_FILE}")
    print("----------------------------------\n")

    print("Loading GSM8K train split...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    subset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

    print("Building prompts...")
    prompts = []
    ground_truths = []
    questions = []

    for ex in subset:
        question = ex["question"]
        gt = extract_ground_truth(ex["answer"])

        prompts.append(build_prompt(question))
        ground_truths.append(gt)
        questions.append(question)

    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=NUM_GPUS,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    print("Generating trajectories in batches...\n")

    total_processed = 0
    total_verified = 0

    with open(OUTPUT_FILE, "a") as fout:
        for start in range(0, len(prompts), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(prompts))
            batch_prompts = prompts[start:end]

            print(f"Processing examples {start} to {end - 1} ...")
            outputs = llm.generate(batch_prompts, sampling_params)

            for local_idx, output in enumerate(outputs):
                global_idx = start + local_idx
                generated_text = output.outputs[0].text.strip()
                assistant_content = (ASSISTANT_PREFIX + generated_text).strip()

                model_answer = extract_model_answer(generated_text)
                gt = ground_truths[global_idx]

                total_processed += 1

                if model_answer is not None and gt is not None and model_answer == gt:
                    record = {
                        "id": global_idx,
                        "messages": [
                            {"role": "user", "content": questions[global_idx]},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        "ground_truth": gt,
                        "model_answer": model_answer,
                        "verified": True,
                    }
                    fout.write(json.dumps(record) + "\n")
                    total_verified += 1

            fout.flush()
            print(
                f"Done through {end - 1}. "
                f"Verified so far: {total_verified}/{total_processed} "
                f"({100.0 * total_verified / max(total_processed, 1):.2f}%)\n"
            )

    print("==========================================")
    print(f"Finished!")
    print(f"Verified trajectories: {total_verified}/{len(prompts)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("==========================================")


if __name__ == "__main__":
    main()