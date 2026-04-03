import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import transformers
import transformers.activations
from datasets import load_dataset

# Patch for transformers compatibility
if not hasattr(transformers.activations, "PytorchGELUTanh"):
    transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

# ---------------- CONFIG ----------------
model_path = os.path.expanduser(
    "~/scratch/araj72/agentDistillation/qwen_0.5b_sft_midway_h200"
)

max_seq_length = 2048
num_eval = 100

# ---------------- LOAD MODEL ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# ---------------- DATA ----------------
dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_eval))

def extract_answer(text):
    text = text.replace(",", "")

    boxed_matches = re.findall(r'\\boxed\{(-?\d+(?:\.\d+)?)\}', text)
    if boxed_matches:
        return boxed_matches[-1]

    explicit = re.findall(r'[Tt]he answer is:?\s*(-?\d+(?:\.\d+)?)', text)
    if explicit:
        return explicit[-1]

    all_numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return all_numbers[-1] if all_numbers else None

correct = 0
print("Evaluating SFT student...")

for i, example in enumerate(dataset):
    messages = [{"role": "user", "content": example["question"]}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
        use_cache=True,
    )

    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    prediction = extract_answer(response)
    ground_truth = example["answer"].split("####")[-1].strip().replace(",", "")

    if prediction == ground_truth:
        correct += 1

    if i < 5:
        print(f"\nQuestion: {example['question']}")
        print(f"Model Response: {response}")
        print(f"Extracted: {prediction} | Actual: {ground_truth}")

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{num_eval}... Current Accuracy: {100 * correct / (i+1):.1f}%")

final_acc = 100 * correct / num_eval
print(f"\nFinal SFT Student Accuracy (Baseline 2): {final_acc:.2f}% ({correct}/{num_eval})")