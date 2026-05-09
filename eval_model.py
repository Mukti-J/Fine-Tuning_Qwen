#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Security Models
Metrics: Accuracy, Precision, Recall, F1
Sequential inference (stable) + clean table output + ROCm-friendly
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
    model.eval()  # ✅ Critical: set eval mode
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # ✅ ROCm/decoder-only fix: left-padding + pad token
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_eval_data():
    print(f"Loading eval data from {EVAL_DATA}...")
    data = []
    with open(EVAL_DATA, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_label(response):
    """Extract is_vulnerable label from JSON or fallback string parse"""
    try:
        result = json.loads(response)
        val = result.get("is_vulnerable")
        if isinstance(val, bool):
            return val
    except:
        pass
    # Fallback: string match
    resp_lower = response.lower()
    if '"is_vulnerable": false' in resp_lower or '"is_vulnerable":false' in resp_lower:
        return False
    if '"is_vulnerable": true' in resp_lower or '"is_vulnerable":true' in resp_lower:
        return True
    return None

def print_eval_table(metrics, eval_runtime, total_tokens, total_samples):
    """Print clean, readable table output"""
    samples_sec = total_samples / eval_runtime if eval_runtime > 0 else 0
    tokens_sec = total_tokens / eval_runtime if eval_runtime > 0 else 0
    
    print("\n" + "═"*60)
    print(f"{'EVALUATION RESULTS':^60}")
    print("═"*60)
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Eval Data:   {EVAL_DATA}")
    print(f"  Samples:     {total_samples}")
    print("─"*60)
    print(f"  {'Metric':<15} {'Value':>12}")
    print("─"*60)
    print(f"  {'Accuracy':<15} {metrics['accuracy']:>12.4f}")
    print(f"  {'Precision':<15} {metrics['precision']:>12.4f}")
    print(f"  {'Recall':<15} {metrics['recall']:>12.4f}")
    print(f"  {'F1 Score':<15} {metrics['f1']:>12.4f}")
    print("─"*60)
    print(f"  {'Confusion Matrix':^28}")
    print(f"  TP: {metrics['true_positive']:>4}  |  FP: {metrics['false_positive']:>4}")
    print(f"  TN: {metrics['true_negative']:>4}  |  FN: {metrics['false_negative']:>4}")
    print("─"*60)
    print(f"  {'Runtime':<15} {eval_runtime:>11.2f}s")
    print(f"  {'Samples/sec':<15} {samples_sec:>12.2f}")
    print(f"  {'Tokens/sec':<15} {tokens_sec:>12.2f}")
    print("═"*60 + "\n")
    
    return {"samples_per_sec": round(samples_sec, 2), "tokens_per_sec": round(tokens_sec, 2)}

def evaluate(model, tokenizer, eval_data):
    print(f"Evaluating {len(eval_data)} samples (sequential, optimized)...\n")
    
    # ✅ Optimasi 1: Set generation config sekali di awal
    model.generation_config.max_new_tokens = 64  # Cukup untuk JSON singkat
    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token else tokenizer.eos_token_id
    
    # ✅ Optimasi 2: Pastikan model di GPU + eval mode
    model.eval()
    if hasattr(torch, 'hip'):
        torch.hip.empty_cache()
    
    tp = fp = tn = fn = 0
    results = []
    total_tokens = 0
    eval_start = time.time()

    for idx, item in enumerate(tqdm(eval_data, desc="Inference")):
        messages = item["messages"]
        prompt = tokenizer.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # ✅ Optimasi 3: Tokenize + move to device sekali
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # ✅ Optimasi 4: Hapus param invalid, pakai generation_config
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # ❌ HAPUS: temperature=0.1, top_p=..., top_k=... (invalid saat do_sample=False)
            )
        
        # Decode hanya generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        total_tokens += len(generated_ids)
        
        # Extract labels
        true_label = extract_label(messages[-1]["content"])
        pred_label = extract_label(response)
        
        results.append({
            "idx": idx,
            "prompt_preview": messages[1]["content"][:150] + "..." if len(messages) > 1 else "",
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "true_label": true_label,
            "predicted_label": pred_label,
            "parsed_ok": pred_label is not None
        })
        
        # Update confusion matrix
        if true_label is not None and pred_label is not None:
            if pred_label and true_label: tp += 1
            elif pred_label and not true_label: fp += 1
            elif not pred_label and not true_label: tn += 1
            elif not pred_label and true_label: fn += 1
        
        # ✅ Optimasi 5: Clear cache tiap 200 samples (bukan tiap iterasi)
        if (idx + 1) % 200 == 0 and hasattr(torch, 'hip'):
            torch.hip.empty_cache()

    # ✅ Optimasi 6: Sync sekali di akhir
    if hasattr(torch, 'hip'):
        torch.hip.synchronize()
    
    eval_runtime = time.time() - eval_start
    
    # Compute metrics
    total_valid = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "accuracy": round((tp + tn) / total_valid, 4) if total_valid > 0 else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "total_samples": len(eval_data),
        "parse_success_rate": round(sum(1 for r in results if r["parsed_ok"]) / len(results), 4)
    }
    
    return metrics, results, eval_runtime, total_tokens

def main():
    model, tokenizer = load_model()
    eval_data = load_eval_data()
    
    metrics, results, runtime, total_tokens = evaluate(model, tokenizer, eval_data)
    speed_info = print_eval_table(metrics, runtime, total_tokens, len(eval_data))
    
    # Save full report
    output = {
        "model_path": MODEL_PATH,
        "eval_data": EVAL_DATA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "performance": {
            "eval_runtime_sec": round(runtime, 2),
            **speed_info
        },
        "sample_results": results[:10]  # First 10 for debugging
    }
    
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved: {OUTPUT_RESULTS}")
    
    # ✅ Quick pass/fail hint
    if metrics["f1"] >= 0.85 and metrics["parse_success_rate"] >= 0.92:
        print("🎯 Model meets deployment thresholds → Ready for merge & deploy")
    else:
        print("⚠️  Model below thresholds → Review training or data")

if __name__ == "__main__":
    main()