import json
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
# Replace with your chosen Hugging Face model ID
MODEL_ID = "txn545/Qwen3.5-122B-A10B-NVFP4" # Using 72B as a more manageable HPC example
NUM_EXAMPLES = 500
NUM_GPUS = 4 # Must match your SLURM --gres=gpu:X count

def extract_ground_truth(answer_str):
    parts = answer_str.split("####")
    if len(parts) == 2:
        return parts[1].strip().replace(",", "")
    return None

def extract_model_answer(text):
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

def main():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    subset = dataset.select(range(NUM_EXAMPLES))

    # Format the prompts
    prompts = []
    ground_truths = []
    for example in subset:
        prompts.append(f"Question: {example['question']}\nAnswer: Let's think step by step.")
        ground_truths.append(extract_ground_truth(example['answer']))

    print(f"Initializing vLLM Engine across {NUM_GPUS} GPUs...")
    # tensor_parallel_size automatically shards the model across your GPUs
    llm = LLM(model=MODEL_ID, tensor_parallel_size=NUM_GPUS, trust_remote_code=True)
    
    # Configure generation parameters
    sampling_params = SamplingParams(temperature=0.3, max_tokens=1000)

    print(f"Generating trajectories for {NUM_EXAMPLES} examples...")
    # vLLM handles the massive batch processing automatically
    outputs = llm.generate(prompts, sampling_params)

    golden_trajectories = []
    
    # Verify outputs
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        model_answer = extract_model_answer(generated_text)
        
        # Keep only if the math is correct
        if model_answer and model_answer == ground_truths[i]:
            golden_trajectories.append({
                "messages": [
                    {"role": "user", "content": prompts[i]},
                    {"role": "assistant", "content": generated_text}
                ]
            })

    # Save to JSONL
    output_file = "data/gsm8k_hf_golden_trajectories.jsonl"
    with open(output_file, 'w') as f:
        for traj in golden_trajectories:
            f.write(json.dumps(traj) + '\n')

    print(f"\nGeneration Complete!")
    print(f"Successfully verified {len(golden_trajectories)} / {NUM_EXAMPLES} golden trajectories.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()