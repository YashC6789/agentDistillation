import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import re

# 1. Load the model you just trained
model_path = os.path.expanduser("~/scratch/araj72/agentDistillation/qwen_0.5b_sft_midway")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = False,
)
FastLanguageModel.for_inference(model)

# 2. Load GSM8K Test set
dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(100)) # Small sample for speed

def extract_answer(text):
    # 1. Try to find numbers inside \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{([\d,]+)\}', text)
    if boxed_matches:
        return boxed_matches[-1].replace(",", "")
    
    # 2. Try the original "The answer is: " format
    matches = re.findall(r'[Tt]he answer is:?\s*([\d,]+)', text)
    if matches:
        return matches[-1].replace(",", "")

    # 3. Fallback: Just grab the last number in the text
    # (Math models often end with the answer)
    all_numbers = re.findall(r'\d+', text.replace(",", ""))
    return all_numbers[-1] if all_numbers else None

correct = 0
print("Evaluating SFT Student...")
for i, example in enumerate(dataset):
    messages = [{"role": "user", "content": example['question']}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    prediction = extract_answer(response)
    ground_truth = example['answer'].split("####")[-1].strip()
    
    if prediction == ground_truth:
        correct += 1

        # Add this inside your loop in eval_sft.py
    if i < 5:
        print(f"\nQuestion: {example['question']}")
        print(f"Model Response: {response}")
        print(f"Extracted: {prediction} | Actual: {ground_truth}")
    
    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/100... Current Accuracy: {correct/(i+1)*100:.1f}%")

print(f"\nFinal SFT Student Accuracy (Baseline 2): {correct}%")