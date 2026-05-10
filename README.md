# Apache License
Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

# Fine-Tuning Qwen2.5-Coder-14B for Code Security

This repository contains scripts for fine-tuning **Qwen/Qwen2.5-Coder-14B-Instruct** using LoRA (Low-Rank Adaptation) for two security tasks:

- **Auditor Model**: Vulnerability detection
- **Builder Model**: Secure code remediation/fixing

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Merging LoRA Adapter](#merging-lora-adapter)
6. [Evaluation](#evaluation)
7. [Example Prompts](#example-prompts)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Requirements

### Hardware
- GPU: NVIDIA A100/H100 (40GB+ VRAM) or equivalent ROCm GPU
- RAM: 64GB+ system memory
- Storage: 100GB+ free space for checkpoints

### Software
- Python 3.10+
- CUDA 12.1+ or ROCm 7.0+
- PyTorch 2.4+

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Fine-Tuning_Qwen.git
cd Fine-Tuning_Qwen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch==2.4.0
pip install transformers>=4.45.0
pip install peft>=0.13.0
pip install datasets>=3.0.0
pip install accelerate>=0.34.0
pip install pandas>=2.0.0
pip install tqdm>=4.66.0
```

---

## Dataset Preparation

### Option 1: Use Preprocessing Script

If you have raw CSV data, use `preprocess_final.py`:

```bash
python preprocess_final.py \
    --input ./data/raw/code-security-dataset.csv \
    --output ./data/processed/
```

Expected CSV columns:
- `code`: Source code
- `code_fixed`: Fixed code (for Builder)
- `is_vulnerable`: Boolean flag
- `cwe_id`: CWE identifier
- `severity`: Critical/High/Medium/Low

### Option 2: Manual JSONL Format

Create JSONL files with ChatML format:

**For Auditor** (`./data/detection_tasks.jsonl`):
```json
{"instruction": "Analyze the following code for vulnerabilities", "input": "function vulnerable() { eval(userInput); }", "output": "{\"is_vulnerable\": true, \"cwe_id\": 94, \"owasp_category\": \"A03:2021-Injection\", \"severity\": \"critical\", \"description\": \"Code injection via eval()\"}"}
```

**For Builder** (`./data/fixing_tasks.jsonl`):
```json
{"instruction": "Fix the security vulnerability in this code", "input": "function vulnerable() { eval(userInput); }", "output": "```javascript\nfunction safe() { // Use safe parsing instead of eval }\n```"}
```

---

## Training

### Training Auditor Model

```bash
python scripts/train_lora_minimal.py
```

Key parameters in `scripts/train_lora_minimal.py`:
| Parameter | Value | Description |
|-----------|-------|-------------|
| MODEL_NAME | Qwen/Qwen2.5-Coder-14B-Instruct | Base model |
| DATA_PATH | ./data/detection_tasks.jsonl | Training data |
| OUTPUT_DIR | ./outputs/auditor_lora | LoRA adapter output |
| MERGED_DIR | ./models/auditor_merged | Final merged model |
| LoRA r | 64 | Rank dimension |
| LoRA alpha | 128 | Scaling factor |
| batch_size | 4 | Per-device batch size |
| gradient_accumulation | 8 | Effective batch = 32 |
| learning_rate | 2e-4 | AdamW learning rate |
| epochs | 3 | Training epochs |
| max_length | 4096 | Sequence length |

### Training Builder Model

```bash
python scripts/train_builder_minimal.py
```

Key parameters in `scripts/train_builder_minimal.py`:
| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 6 | Per-device batch size |
| gradient_accumulation | 6 | Effective batch = 36 |
| learning_rate | 1.5e-4 | Slightly lower LR |
| max_length | 2048 | Shorter sequences |

### Training with Custom Paths

Edit the script variables:
```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
DATA_PATH = "./your/custom/path/detection_tasks.jsonl"
OUTPUT_DIR = "./your/output/auditor_lora"
MERGED_DIR = "./your/models/auditor_merged"
```

---

## Merging LoRA Adapter

After training, merge the LoRA adapter with the base model:

```python
# Already included in training scripts at the end:
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(base_model, "./outputs/auditor_lora")
merged = peft_model.merge_and_unload()
merged.save_pretrained("./models/auditor_merged", safe_serialization=True)
```

Or use the merge script separately:

```bash
python scripts/merge_lora.py \
    --adapter ./outputs/auditor_lora \
    --output ./models/auditor_merged
```

---

## Evaluation

Run evaluation on test dataset:

```bash
python eval_model.py \
    --model_path ./models/auditor_merged \
    --test_data ./data/test_detection.jsonl \
    --output ./results/eval_results.json
```

---

## Example Prompts

### Auditor: Vulnerability Detection

**Prompt:**
```
Analyze the following code for security vulnerabilities:

```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
```

Determine if there are vulnerabilities, and if so, provide the details in JSON format.
```

**Expected Output:**
```json
{
  "is_vulnerable": true,
  "cwe_id": 89,
  "owasp_category": "A03:2021-Injection",
  "severity": "critical",
  "line_location": {"start": 2, "end": 2},
  "description": "SQL Injection vulnerability due to string interpolation in query",
  "confidence": 0.95
}
```

### Builder: Secure Code Remediation

**Prompt:**
```
Fix the security vulnerability in the following code:

```python
import os
filename = input("Enter filename: ")
os.system(f"cat {filename}")
```

Provide the fixed code with proper security measures.
```

**Expected Output:**
```python
import subprocess
import shlex

filename = input("Enter filename: ")
safe_filename = shlex.quote(filename)
result = subprocess.run(['cat', safe_filename], capture_output=True, text=True)
print(result.stdout)
```

### Safe Code Validation

**Prompt:**
```
Analyze this code for vulnerabilities:

```python
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

Is this code secure? Provide details if any issues found.
```

**Expected Output:**
```json
{
  "is_vulnerable": false,
  "description": "Code is secure. Email validation uses proper regex pattern without any injection risks.",
  "confidence": 0.98
}
```

---

## Configuration Reference

### LoRA Configuration

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `r` (rank) | 64 | Higher = more capacity, more VRAM |
| `lora_alpha` | 128 | Usually 2x rank |
| `lora_dropout` | 0.05 | Regularization |
| `target_modules` | all-linear | Apply to all linear layers |

### Training Configuration

| Parameter | Auditor | Builder | Notes |
|-----------|---------|---------|-------|
| `per_device_train_batch_size` | 4 | 6 | Adjust for VRAM |
| `gradient_accumulation_steps` | 8 | 6 | Effective batch = batch × grad_accum |
| `learning_rate` | 2e-4 | 1.5e-4 | Builder uses lower LR |
| `num_train_epochs` | 3 | 3 | |
| `max_length` | 4096 | 2048 | Auditor needs longer context |
| `warmup_ratio` | 0.05 | 0.05 | 5% of steps |
| `weight_decay` | 0.01 | 0.01 | AdamW decay |
| `lr_scheduler_type` | cosine | cosine | |
| `fp16` | true | true | Mixed precision |

---

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size:**
   ```python
   per_device_train_batch_size=2
   gradient_accumulation_steps=16
   ```

2. **Enable gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Use LoRA with smaller rank:**
   ```python
   lora_config = LoraConfig(r=32, lora_alpha=64, ...)
   ```

### ROCm Issues (AMD GPU)

If using ROCm 7.0, set environment variables:

```python
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False"
```

### DataCollator Warning

If you see `DataCollator` warnings, this is normal. The scripts handle it with:
```python
remove_unused_columns=False
dataloader_num_workers=0
```

### Slow Training

1. Use `bf16` instead of `fp16` on supported GPUs
2. Enable `gradient_checkpointing`
3. Use `torch.compile()` if on PyTorch 2.4+

---

## Project Structure

```
Fine-Tuning_Qwen/
├── README.md
├── preprocess_final.py          # Dataset preprocessing
├── eval_model.py                 # Model evaluation
├── configs/
│   ├── auditor_config.yaml       # Auditor training config
│   └── builder_config.yaml       # Builder training config
├── scripts/
│   ├── train_lora_minimal.py    # Auditor training script
│   ├── train_builder_minimal.py # Builder training script
│   └── server.py                 # Inference server
├── data/
│   ├── detection_tasks.jsonl     # Auditor dataset
│   └── fixing_tasks.jsonl        # Builder dataset
├── outputs/
│   ├── auditor_lora/             # Auditor LoRA adapter
│   └── builder_lora/             # Builder LoRA adapter
└── models/
    ├── auditor_merged/           # Merged Auditor model
    └── builder_merged/           # Merged Builder model
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) by Alibaba
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
- [Transformers](https://github.com/huggingface/transformers) by Hugging Face