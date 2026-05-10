---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
pipeline_tag: text-generation
tags:
  - code-generation
  - secure-coding
  - patch-generation
  - rocm
  - qwen2.5
  - amd-hackathon
---

# 🔧 Security Builder Model (14B)

Fine-tuned Qwen2.5-Coder-14B-Instruct khusus untuk **generasi patch keamanan & penulisan kode aman**. Melengkapi Auditor model dengan mengubah laporan kerentanan menjadi kode perbaikan yang production-ready.

## 🚀 Quick Load
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "lablab-ai-amd-developer-hackathon/security-builder-14b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

###  💬 Example Usage (JSON Mode)
messages = [
    {"role": "user", "content": "Fix the buffer overflow and return JSON with keys: fixed_code, explanation, cwe_mitigated."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

import json
print(json.loads(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)))
```
#### 🛠️ Technical Specifications

| Parameter | Value |
| :--- | :--- |
| **Base Model** | Qwen2.5-Coder-14B-Instruct |
| **Fine-tuning** | LoRA (r=64, alpha=128, dropout=0.05) |
| **Training Data** | Custom secure coding & patch dataset |
| **Epochs** | 3 |
| **Precision** | float16 (ROCm-optimized) |
| **Format** | Safetensors (6 shards, ~28GB) |
| **VRAM Required** | ~38-42 GB |

##### 🖥️ ROCm & Hardware Optimization
Dioptimalkan untuk AMD Instinct MI300X / ROCm 7.0. Disarankan set env var berikut sebelum inference:
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False

###### 🔌 API Integration
Designed for CI/CD integration. Gunakan response_format={"type":"json_object"} untuk parsing otomatis patch & metadata keamanan.

####### 📜 License & Credits
Apache 2.0. Developed for the AMD Developer Hackathon 2026.
