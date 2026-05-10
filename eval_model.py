#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned Security Models (Merged)
Metrics: Accuracy, Precision, Recall, F1
Sequential inference (stable) + clean table output + ROCm-friendly
"""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["auditor", "builder"], required=True)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference (default: 8)")
args = parser.parse_args()

# ✅ Paths updated: merged models (no LoRA adapter needed)
BASE_DIR = "/home/security-ft/code-security-ft/models"
MODEL_PATHS = {
    "auditor": f"{BASE_DIR}/auditor_merged",
    "builder": f"{BASE_DIR}/builder_merged"
}

# ✅ Eval data paths — sesuaikan jika berbeda
EVAL_DATA_PATHS = {
    "auditor": "/mnt/scratch/processed/auditor_eval.jsonl",
    "builder": "/mnt/scratch/processed/builder_eval.jsonl"
}

MODEL_PATH = MODEL_PATHS[args.model]
EVAL_DATA  = EVAL_DATA_PATHS[args.model]

# ✅ Output results disimpan di samping model merged
OUTPUT_RESULTS = f"{BASE_DIR}/{args.model}_merged_eval_results.json"


def load_model():
    """Load merged model langsung — tidak butuh PeftModel lagi."""
    print(f"Loading merged model from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # ✅ ROCm/decoder-only fix: left-padding + pad token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Device map : {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    print(f"  Dtype      : {next(model.parameters()).dtype}")
    return model, tokenizer


def load_eval_data():
    print(f"Loading eval data from {EVAL_DATA}...")
    data = []
    with open(EVAL_DATA, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} samples.")
    return data


def extract_label(response):
    """Extract is_vulnerable label from JSON or fallback string parse."""
    try:
        result = json.loads(response)
        val = result.get("is_vulnerable")
        if isinstance(val, bool):
            return val
    except Exception:
        pass
    # Fallback: string match
    resp_lower = response.lower()
    if '"is_vulnerable": false' in resp_lower or '"is_vulnerable":false' in resp_lower:
        return False
    if '"is_vulnerable": true' in resp_lower or '"is_vulnerable":true' in resp_lower:
        return True
    return None


def print_eval_table(metrics, eval_runtime, total_tokens, total_samples):
    """Print clean, readable table output."""
    samples_sec = total_samples / eval_runtime if eval_runtime > 0 else 0
    tokens_sec  = total_tokens  / eval_runtime if eval_runtime > 0 else 0

    print("\n" + "═" * 60)
    print(f"{'EVALUATION RESULTS':^60}")
    print("═" * 60)
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Eval Data:   {EVAL_DATA}")
    print(f"  Samples:     {total_samples}")
    print("─" * 60)
    print(f"  {'Metric':<15} {'Value':>12}")
    print("─" * 60)
    print(f"  {'Accuracy':<15} {metrics['accuracy']:>12.4f}")
    print(f"  {'Precision':<15} {metrics['precision']:>12.4f}")
    print(f"  {'Recall':<15} {metrics['recall']:>12.4f}")
    print(f"  {'F1 Score':<15} {metrics['f1']:>12.4f}")
    print("─" * 60)
    print(f"  {'Confusion Matrix':^28}")
    print(f"  TP: {metrics['true_positive']:>4}  |  FP: {metrics['false_positive']:>4}")
    print(f"  TN: {metrics['true_negative']:>4}  |  FN: {metrics['false_negative']:>4}")
    print("─" * 60)
    print(f"  {'Parse Success':<15} {metrics['parse_success_rate']:>12.4f}")
    print(f"  {'Runtime':<15} {eval_runtime:>11.2f}s")
    print(f"  {'Samples/sec':<15} {samples_sec:>12.2f}")
    print(f"  {'Tokens/sec':<15} {tokens_sec:>12.2f}")
    print("═" * 60 + "\n")

    return {"samples_per_sec": round(samples_sec, 2), "tokens_per_sec": round(tokens_sec, 2)}


def evaluate(model, tokenizer, eval_data):
    batch_size = args.batch_size
    print(f"Evaluating {len(eval_data)} samples in batches of {batch_size}...\n")

    model.generation_config.max_new_tokens = 64
    model.generation_config.do_sample      = False
    model.generation_config.pad_token_id   = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    model.eval()
    if hasattr(torch, "hip"):
        torch.hip.empty_cache()

    tp = fp = tn = fn = 0
    results      = []
    total_tokens = 0
    eval_start   = time.time()

    num_batches = (len(eval_data) + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(
        tqdm(range(0, len(eval_data), batch_size), desc="Batch", total=num_batches)
    ):
        batch_data = eval_data[batch_start : batch_start + batch_size]

        # Build prompts — strip assistant turns, keep only user/system
        prompts = [
            tokenizer.apply_chat_template(
                [m for m in item["messages"] if m["role"] != "assistant"],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch_data
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        for i, output in enumerate(outputs):
            item          = batch_data[i]
            generated_ids = output[inputs["input_ids"].shape[1]:]
            response      = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            total_tokens += len(generated_ids)

            # Ground-truth label is the last assistant message
            true_label = extract_label(item["messages"][-1]["content"])
            pred_label = extract_label(response)

            results.append({
                "idx":            batch_start + i,
                "prompt_preview": (
                    item["messages"][1]["content"][:150] + "..."
                    if len(item["messages"]) > 1 else ""
                ),
                "response_preview": response[:200] + ("..." if len(response) > 200 else ""),
                "true_label":      true_label,
                "predicted_label": pred_label,
                "parsed_ok":       pred_label is not None,
            })

            if true_label is not None and pred_label is not None:
                if     pred_label and     true_label: tp += 1
                elif   pred_label and not true_label: fp += 1
                elif not pred_label and not true_label: tn += 1
                elif not pred_label and     true_label: fn += 1

        # Periodic VRAM flush on ROCm
        if (batch_idx + 1) % 25 == 0 and hasattr(torch, "hip"):
            torch.hip.empty_cache()

    if hasattr(torch, "hip"):
        torch.hip.synchronize()

    eval_runtime = time.time() - eval_start

    total_valid = tp + tn + fp + fn
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    parse_ok_count = sum(1 for r in results if r["parsed_ok"])

    metrics = {
        "accuracy":           round((tp + tn) / total_valid, 4) if total_valid > 0 else 0.0,
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
        "true_positive":      tp,
        "false_positive":     fp,
        "true_negative":      tn,
        "false_negative":     fn,
        "total_samples":      len(eval_data),
        "parse_success_rate": round(parse_ok_count / len(results), 4) if results else 0.0,
        "unparseable":        len(results) - parse_ok_count,
    }

    return metrics, results, eval_runtime, total_tokens


def main():
    model, tokenizer = load_model()
    eval_data        = load_eval_data()

    metrics, results, runtime, total_tokens = evaluate(model, tokenizer, eval_data)
    speed_info = print_eval_table(metrics, runtime, total_tokens, len(eval_data))

    output = {
        "model":      args.model,
        "model_path": MODEL_PATH,
        "eval_data":  EVAL_DATA,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics":    metrics,
        "performance": {
            "eval_runtime_sec": round(runtime, 2),
            **speed_info,
        },
        "sample_results": results[:10],   # 10 sampel pertama untuk debugging
    }

    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved: {OUTPUT_RESULTS}")

    # Pass/fail hint
    if metrics["f1"] >= 0.85 and metrics["parse_success_rate"] >= 0.92:
        print("🎯 Model meets deployment thresholds → Ready for production")
    else:
        print("⚠️  Model below thresholds → Review training or data quality")
        print(f"   F1={metrics['f1']:.4f} (need ≥0.85) | "
              f"Parse={metrics['parse_success_rate']:.4f} (need ≥0.92)")


if __name__ == "__main__":
    main()
