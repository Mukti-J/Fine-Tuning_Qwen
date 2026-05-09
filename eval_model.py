#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Security Models
Metrics: Accuracy, Precision, Recall, F1
"""
import json
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

def evaluate(model, tokenizer, eval_data):
    print(f"Evaluating {len(eval_data)} samples...")

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    results = []

    for item in tqdm(eval_data):
        messages = item["messages"]

        prompt = tokenizer.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]

        true_label = messages[-1]["content"]
        true_label = extract_label(true_label)
        pred_label = extract_label(response)

        results.append({
            "prompt": messages[1]["content"][:200],
            "response": response[:300],
            "true_label": true_label,
            "predicted_label": pred_label
        })

        if true_label is not None and pred_label is not None:
            if pred_label == True and true_label == True:
                true_positive += 1
            elif pred_label == True and true_label == False:
                false_positive += 1
            elif pred_label == False and true_label == False:
                true_negative += 1
            elif pred_label == False and true_label == True:
                false_negative += 1

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-10)
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "total_samples": len(eval_data)
    }

    return metrics, results

def main():
    model, tokenizer = load_model()
    model.eval()

    eval_data = load_eval_data()
    metrics, results = evaluate(model, tokenizer, eval_data)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1 Score:  {metrics['f1']}")
    print(f"TP: {metrics['true_positive']} | FP: {metrics['false_positive']} | TN: {metrics['true_negative']} | FN: {metrics['false_negative']}")
    print("="*50)

    output = {
        "model_path": MODEL_PATH,
        "eval_data": EVAL_DATA,
        "metrics": metrics,
        "sample_results": results[:10]
    }

    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_RESULTS}")

if __name__ == "__main__":
    main()