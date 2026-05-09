#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Security Models
Metrics: Accuracy, Precision, Recall, F1
"""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["auditor", "builder"], required=True)
args = parser.parse_args()

MODEL_PATHS = {
    "auditor": "/mnt/scratch/outputs/auditor",
    "builder": "/mnt/scratch/outputs/builder"
}

EVAL_DATA_PATHS = {
    "auditor": "/mnt/scratch/processed/auditor_eval.jsonl",
    "builder": "/mnt/scratch/processed/builder_eval.jsonl"
}

MODEL_PATH = MODEL_PATHS[args.model]
EVAL_DATA = EVAL_DATA_PATHS[args.model]
BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

OUTPUT_RESULTS = f"/mnt/scratch/outputs/{args.model}_eval_results.json"

def load_model():
    print("Loading base model + LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return model, tokenizer

def load_eval_data():
    print(f"Loading eval data from {EVAL_DATA}...")
    data = []
    with open(EVAL_DATA, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_label(response):
    try:
        result = json.loads(response)
        return result.get("is_vulnerable", None)
    except:
        pass

    if "false" in response.lower() or '"is_vulnerable": false' in response.lower():
        return False
    if "true" in response.lower() or '"is_vulnerable": true' in response.lower():
        return True
    return None

def print_eval_summary(metrics, eval_runtime, total_tokens):
    samples_sec = metrics["total_samples"] / eval_runtime if eval_runtime > 0 else 0
    tokens_sec = total_tokens / eval_runtime if eval_runtime > 0 else 0

    bar_len = 50
    print()
    print(f"{'='*bar_len}")
    print(f"{'EVALUATION RESULTS':^{bar_len}}")
    print(f"{'='*bar_len}")
    print()
    print(f"  {'Model':<25} {MODEL_PATH}")
    print(f"  {'Eval Data':<25} {EVAL_DATA}")
    print()
    print(f"{'-'*bar_len}")
    print(f"  {'Metric':<20} {'Value':>12}")
    print(f"{'-'*bar_len}")
    print(f"  {'Accuracy':<20} {metrics['accuracy']:>12.4f}")
    print(f"  {'Precision':<20} {metrics['precision']:>12.4f}")
    print(f"  {'Recall':<20} {metrics['recall']:>12.4f}")
    print(f"  {'F1 Score':<20} {metrics['f1']:>12.4f}")
    print(f"{'-'*bar_len}")
    print()
    print(f"  {'Confusion Matrix':^}")
    print(f"  {'TP (True Positive):':<25} {metrics['true_positive']:>5}")
    print(f"  {'FP (False Positive):':<25} {metrics['false_positive']:>5}")
    print(f"  {'TN (True Negative):':<25} {metrics['true_negative']:>5}")
    print(f"  {'FN (False Negative):':<25} {metrics['false_negative']:>5}")
    print()
    print(f"{'-'*bar_len}")
    print(f"  {'Total Samples':<20} {metrics['total_samples']:>12}")
    print(f"  {'Total Tokens':<20} {total_tokens:>12,}")
    print(f"  {'Eval Runtime':<20} {eval_runtime:>11.2f}s")
    print(f"  {'Samples/sec':<20} {samples_sec:>12.2f}")
    print(f"  {'Tokens/sec':<20} {tokens_sec:>12.2f}")
    print(f"{'='*bar_len}")
    print()

    return {
        "samples_per_sec": round(samples_sec, 2),
        "tokens_per_sec": round(tokens_sec, 2)
    }

def evaluate(model, tokenizer, eval_data, batch_size=32):
    print(f"Evaluating {len(eval_data)} samples (batch_size={batch_size})...")

    tp, fp, tn, fn = 0, 0, 0, 0
    results = []
    total_tokens = 0
    eval_start = time.time()

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch = eval_data[i:i+batch_size]
        prompts = []
        true_labels = []
        
        for item in batch:
            messages = item["messages"]
            prompt = tokenizer.apply_chat_template(
                [m for m in messages if m["role"] != "assistant"],
                tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            true_labels.append(extract_label(messages[-1]["content"]))

        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                attention_mask=inputs["attention_mask"]
            )

        # Decode batch & compute metrics
        for j, out in enumerate(outputs):
            pred_text = tokenizer.decode(out, skip_special_tokens=True)[len(prompts[j]):]
            pred_label = extract_label(pred_text)
            true_label = true_labels[j]
            
            total_tokens += (len(out) - len(inputs["input_ids"][j]))
            results.append({
                "prompt": prompts[j][:200],
                "response": pred_text[:300],
                "true_label": true_label,
                "predicted_label": pred_label
            })

            if true_label is not None and pred_label is not None:
                if pred_label and true_label: tp += 1
                elif pred_label and not true_label: fp += 1
                elif not pred_label and not true_label: tn += 1
                elif not pred_label and true_label: fn += 1

    eval_runtime = time.time() - eval_start
    total_valid = tp + tn + fp + fn
    metrics = {
        "accuracy": round((tp + tn) / (total_valid + 1e-10), 4),
        "precision": round(tp / (tp + fp + 1e-10), 4),
        "recall": round(tp / (tp + fn + 1e-10), 4),
        "f1": round(2 * ((tp/(tp+fp+1e-10)) * (tp/(tp+fn+1e-10))) / ((tp/(tp+fp+1e-10)) + (tp/(tp+fn+1e-10)) + 1e-10), 4),
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "total_samples": len(eval_data),
        "total_tokens": total_tokens,
        "eval_runtime": round(eval_runtime, 2)
    }
    return metrics, results

def main():
    model, tokenizer = load_model()

    eval_data = load_eval_data()
    metrics, results = evaluate(model, tokenizer, eval_data)

    speed_info = print_eval_summary(metrics, metrics["eval_runtime"], metrics["total_tokens"])

    output = {
        "model_path": MODEL_PATH,
        "eval_data": EVAL_DATA,
        "metrics": metrics,
        "performance": speed_info,
        "sample_results": results[:10]
    }

    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_RESULTS}")

if __name__ == "__main__":
    main()  