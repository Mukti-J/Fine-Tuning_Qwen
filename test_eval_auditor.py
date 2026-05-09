import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
ADAPTER_PATH = "/mnt/scratch/outputs/auditor"
TEST_SET = "/mnt/scratch/data/auditor_test.jsonl"
OUTPUT_REPORT = "/mnt/scratch/reports/auditor_eval.json"

def extract_json(text):
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL)
    return json.loads(match.group(0)) if match else None

def localize_tolerance(pred_loc, true_loc, tol_lines=1):
    if not pred_loc or not true_loc: return False
    p_line = int(re.search(r'line[:\s]*(\d+)', str(pred_loc), re.I).group(1))
    t_line = int(re.search(r'line[:\s]*(\d+)', str(true_loc), re.I).group(1))
    return abs(p_line - t_line) <= tol_lines

def run_eval():
    print("⏳ Loading tokenizer & model...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    results, parse_count, local_correct = [], 0, 0
    y_true, y_pred = [], []

    print("🚀 Running batch inference...")
    with open(TEST_SET) as f:
        samples = [json.loads(l) for l in f.readlines()]

    for s in samples:
        prompt = tok.apply_chat_template([{"role": "user", "content": s["prompt"]}], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=False)
        out_text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        parsed = extract_json(out_text)
        y_true.append(s["has_vuln"])
        y_pred.append(1 if parsed and parsed.get("has_vulnerability") else 0)
        
        if parsed:
            parse_count += 1
            if localize_tolerance(parsed.get("location"), s.get("true_location")):
                local_correct += 1
        results.append({"input": s["code"][:50]+"...", "output": out_text[:100]+"...", "parsed": bool(parsed)})

    report = {
        "json_parse_rate": parse_count / len(samples),
        "localization_accuracy": local_correct / parse_count if parse_count else 0,
        "detection_metrics": {
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0)
        },
        "samples_evaluated": len(samples)
    }
    Path(OUTPUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_REPORT).write_text(json.dumps(report, indent=2))
    print(f"✅ Report saved: {OUTPUT_REPORT}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_eval()