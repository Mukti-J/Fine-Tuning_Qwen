---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
pipeline_tag: text-generation
tags:
  - code-audit
  - security
  - cwe
  - rocm
  - qwen2.5
  - amd-hackathon
---

# 🔐 Security Auditor Model (14B)

Fine-tuned Qwen2.5-Coder-14B-Instruct khusus untuk **analisis kerentanan keamanan kode**. Model ini mendeteksi vulnerability, mengklasifikasikan CWE ID, menilai severity, dan memberikan rekomendasi mitigasi terstruktur.

## 🚀 Quick Load
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "lablab-ai-amd-developer-hackathon/security-auditor-14b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

💬 Example Usage
messages = [
    {"role": "user", "content": "Audit this C code for security issues:\n\n<code>\nvoid foo(char* buf, char* input) { strcpy(buf, input); }\n</code>"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, temperature=0.2)

print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
