#!/usr/bin/env python3
import json, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/mnt/scratch/outputs/auditor"
BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
EVAL_DATA = "/mnt/scratch/processed/auditor_eval.jsonl"

# Load model
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, MODEL_PATH).eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# Load 100 samples
with open(EVAL_DATA) as f:
    samples = [json.loads(l) for l in f][:100]

# Run inference
correct, parsed = 0, 0
for s in samples:
    prompt = tokenizer.apply_chat_template([m for m in s["messages"] if m["role"]!="assistant"], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        pred = json.loads(resp).get("is_vulnerable")
        true = json.loads(s["messages"][-1]["content"]).get("is_vulnerable")
        if pred is not None and true is not None and pred == true: correct += 1
        parsed += 1
    except: pass

acc = correct / parsed if parsed else 0
print(f"\n✅ Smoke Test (n=100): Accuracy={acc:.4f}, ParseRate={parsed/100:.4f}")
print("🎯 PASS" if acc >= 0.85 and parsed/100 >= 0.92 else "⚠️  FAIL")