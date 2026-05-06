import json
import re
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
# Use the base Instruct model (no AWQ)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
NUM_EXAMPLES = 7473
NUM_GPUS = 2 

def extract_ground_truth(answer_str):
    parts = answer_str.split("####")
    if len(parts) == 2:
        return parts[1].strip().replace(",", "")
    return None

def extract_model_answer(text):
    text = text.replace(",", "")
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1]
    return None

def main():
    # Use expanduser to automatically resolve '~/scratch' to the correct PACE mount
    project_root = os.path.expanduser("~/scratch/araj72/agentDistillation")
    scratch_path = os.path.join(project_root, "hf_cache")
    output_file = os.path.join(project_root, "gsm8k_golden_trajectories_new.jsonl")

    # 1. Ensure the directories exist
    os.makedirs(scratch_path, exist_ok=True)
    
    # 2. Set environment variables so vLLM and HF use the scratch space
    os.environ["HF_HOME"] = scratch_path
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(project_root, "vllm_cache")

    print(f"--- Project Configuration ---")
    print(f"Project Root: {project_root}")
    print(f"Output File:  {output_file}")
    print(f"Model:        {MODEL_ID}")
    print(f"----------------------------")

    # 3. Load GSM8K dataset
    print(f"Loading GSM8K and initializing vLLM on {NUM_GPUS} V100 GPUs...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    subset = dataset.select(range(NUM_EXAMPLES))

    prompts = []
    ground_truths = []
    for i in range(len(subset)):
        # Correctly formatted ChatML for Qwen-2.5-Instruct
        p = f"<|im_start|>user\n{subset[i]['question']}\nShow your work and end with 'The answer is: [number]'.<|im_end|>\n<|im_start|>assistant\nLet's think step by step."
        prompts.append(p)
        ground_truths.append(extract_ground_truth(subset[i]['answer']))

    # 4. Initialize LLM Engine
    # Using float16 for V100 compatibility and 90% memory utilization
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=NUM_GPUS, 
        trust_remote_code=True,
        dtype="float16", 
        gpu_memory_utilization=0.90 
    )
    
    sampling_params = SamplingParams(
        temperature=0.3, 
        max_tokens=1024, 
        stop=["<|im_end|>", "<|im_start|>"]
    )

    # 5. Generate and Verify
    print("Generating trajectories...")
    outputs = llm.generate(prompts, sampling_params)

    golden_trajectories = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        model_answer = extract_model_answer(generated_text)
        
        # Verify the math before adding to our 'Golden' set
        if model_answer and model_answer == ground_truths[i]:
            golden_trajectories.append({
                "messages": [
                    {"role": "user", "content": subset[i]['question']},
                    {"role": "assistant", "content": "Let's think step by step." + generated_text}
                ]
            })

    # 6. Save to JSONL
    with open(output_file, 'w') as f:
        for traj in golden_trajectories:
            f.write(json.dumps(traj) + '\n')

    print(f"\nSuccess! Verified {len(golden_trajectories)}/{NUM_EXAMPLES} trajectories.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()