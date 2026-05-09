#!/usr/bin/env python3
"""
Final Preprocessing Pipeline: CSV → Clean ChatML JSONL (Auditor & Builder)
Optimized for Axolotl + Qwen2.5-Coder-14B
"""
import pandas as pd
import re
import json
import os
from tqdm import tqdm

# === CONFIG ===
INPUT_CSV = "/mnt/scratch/raw/code-security-vulnerability-dataset-final.csv"  # Ganti dengan path file Anda
OUTPUT_DIR = "/mnt/scratch/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"📊 Raw rows: {len(df)}")

# === 1. CLEANING & STANDARDIZATION ===
# Drop baris tanpa kode
df = df.dropna(subset=["code"]).reset_index(drop=True)

# Standardize is_vulnerable
df["is_vulnerable"] = df["is_vulnerable"].astype(str).str.lower().isin(["true", "1", "yes"])

# Extract numeric CWE ID (e.g., "CWE-119" → 119)
def clean_cwe_id(val):
    if pd.isna(val) or str(val).lower() in ["safe", "nan", "none", ""]:
        return None
    match = re.search(r'(\d+)', str(val))
    return int(match.group(1)) if match else None
df["cwe_id_clean"] = df["cwe_id"].apply(clean_cwe_id)

# CWE → OWASP Top 10 2021 Mapping
CWE_TO_OWASP = {
    119: "A03:2021-Injection", 79: "A03:2021-XSS", 89: "A03:2021-SQLi",
    78: "A03:2021-OS Command Injection", 22: "A01:2021-Broken Access Control",
    306: "A04:2021-Insecure Design", 352: "A07:2021-Auth Failures",
    434: "A08:2021-Software & Data Integrity", 502: "A08:2021-Deserialization",
    611: "A04:2021-XXE", 918: "A01:2021-SSRF"
}
df["owasp_mapped"] = df["cwe_id_clean"].map(CWE_TO_OWASP)

# Standardize Severity
def clean_severity(val):
    if pd.isna(val): return "unknown"
    v = str(val).lower().strip()
    if "crit" in v: return "critical"
    if "high" in v: return "high"
    if "med" in v: return "medium"
    if "low" in v: return "low"
    return "unknown"
df["severity_clean"] = df["severity"].apply(clean_severity)

# Handle code_fixed & description
df["code_fixed_clean"] = df["code_fixed"].fillna(df["code"])
df["cwe_desc_clean"] = df["cwe_desc"].apply(
    lambda x: str(x).replace("Description not available", "Potential vulnerability detected.").strip() if pd.notna(x) else "Potential vulnerability detected."
)

# === 2. DATASET SPLIT & BALANCE ===
vuln_df = df[df["is_vulnerable"] == True]
safe_df = df[df["is_vulnerable"] == False]

# Auditor: 50:50 Balance
auditor_safe = safe_df.sample(n=len(vuln_df), random_state=42)
auditor_df = pd.concat([vuln_df, auditor_safe]).sample(frac=1, random_state=42).reset_index(drop=True)

# Builder: Valid fixes + 12% Negative Samples
def is_valid_fix(row):
    if not row["is_vulnerable"]: return False
    diff = len(set(str(row["code"])) ^ set(str(row["code_fixed_clean"])))
    return diff > 15 and str(row["code"]).strip() != str(row["code_fixed_clean"]).strip()

builder_vuln = df[df.apply(is_valid_fix, axis=1)]
builder_neg = safe_df.sample(frac=0.12, random_state=42).reset_index(drop=True)
builder_df = pd.concat([builder_vuln, builder_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# === 3. CHATML CONVERSION ===
def to_chatml(row, task="auditor"):
    # --- AMBIL DATA DENGAN AMAN (Handle NaN) ---
    # Cek apakah CWE valid, jika NaN/None set ke None
    cwe_raw = row.get("cwe_id_clean")
    cwe_val = int(cwe_raw) if pd.notna(cwe_raw) else None
    
    owasp_val = row.get("owasp_mapped")
    if pd.isna(owasp_val): owasp_val = None
    
    sev_val = row.get("severity_clean", "unknown")
    if pd.isna(sev_val): sev_val = "unknown"
    
    desc_val = row.get("cwe_desc_clean", "No description available")
    if pd.isna(desc_val): desc_val = "No description available"
    # --------------------------------------------

    if task == "auditor":
        system = "Anda adalah Senior Application Security Engineer. Analisis kode, deteksi vulnerability, dan outputkan JSON strict. Jika aman, set is_vulnerable=false."
        user = f"```{row.get('language', 'code')}\n{row['code']}\n```\n\nTentukan vulnerability status dan detail."
        assistant = json.dumps({
            "is_vulnerable": bool(row["is_vulnerable"]),
            "cwe_id": cwe_val,  # Sekarang aman dari error int(NaN)
            "owasp_category": owasp_val,
            "severity": sev_val,
            "line_location": {"start": 1, "end": len(str(row['code']).split('\n'))} if row["is_vulnerable"] else None,
            "description": desc_val if row["is_vulnerable"] else "Code is secure.",
            "confidence": 0.90
        })
    else:
        system = "Anda adalah Secure Code Remediation Specialist. Terima kode rentan + konteks CWE/OWASP, lalu hasilkan patch aman. Output hanya kode yang sudah diperbaiki dalam markdown code block."
        vuln_ctx = f"CWE-{cwe_val} | {owasp_val}" if row["is_vulnerable"] else "No vulnerability"
        user = f"```{row.get('language', 'code')}\n{row['code']}\n```\n\nKonteks: {vuln_ctx}\nHasilkan kode yang sudah diperbaiki."
        assistant = f"```{row.get('language', 'code')}\n{row['code_fixed_clean']}\n```"

    return json.dumps({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]})
    

# === 4. SAVE TO JSONL ===
def save_split(df_obj, prefix):
    train, eval = df_obj.iloc[:int(len(df_obj)*0.8)], df_obj.iloc[int(len(df_obj)*0.8):]
    
    for split_name, split_df in [("train", train), ("eval", eval)]:
        filepath = os.path.join(OUTPUT_DIR, f"{prefix}_{split_name}.jsonl")
        with open(filepath, "w") as f:
            for _, row in tqdm(split_df.iterrows(), desc=f"💾 Saving {prefix}_{split_name}"):
                f.write(to_chatml(row, prefix) + "\n")
        print(f"✅ {filepath} | Rows: {len(split_df)}")

print("\n🚀 Converting to ChatML JSONL...")
save_split(auditor_df, "auditor")
save_split(builder_df, "builder")

print("\n🎉 PREPROCESSING COMPLETE. Files ready in /mnt/scratch/processed/")
