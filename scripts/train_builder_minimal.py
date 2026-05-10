import os, json, torch, gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
DATA_PATH = "/home/security-ft/code-security-ft/datasets/fixing_tasks.jsonl"
OUTPUT_DIR = "/home/security-ft/code-security-ft/outputs/builder_lora"
MERGED_DIR = "/home/security-ft/code-security-ft/models/builder_merged"

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False"
os.environ["WANDB_DISABLED"] = "true"

def main():
    print(" Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(" Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)

    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05, target_modules="all-linear", bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(" Processing dataset...")
    data = []
    with open(DATA_PATH) as f:
        for line in f:
            obj = json.loads(line)
            instr, inp, out = obj.get("instruction",""), obj.get("input",""), obj.get("output","")
            user_msg = f"{instr}\n\n<code>\n{inp}\n</code>" if inp else instr
            data.append({"messages": [{"role":"user","content":user_msg}, {"role":"assistant","content":out}]})

    split_idx = int(len(data) * 0.95)
    train_data, val_data = data[:split_idx], data[split_idx:]

    def format_fn(examples):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in examples["messages"]]
        tok = tokenizer(texts, truncation=True, max_length=2048, padding="max_length")
        tok["labels"] = [ids.copy() for ids in tok["input_ids"]]
        return tok

    train_ds = Dataset.from_list(train_data).map(format_fn, batched=True, remove_columns=["messages"])
    val_ds = Dataset.from_list(val_data).map(format_fn, batched=True, remove_columns=["messages"]) if val_data else None

    print("🚀 Starting Builder training...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=6, gradient_accumulation_steps=6,
        learning_rate=1.5e-4, warmup_ratio=0.05, weight_decay=0.01,
        lr_scheduler_type="cosine", fp16=True, logging_steps=10, save_steps=150,
        eval_strategy="steps" if val_ds else "no", eval_steps=150, save_total_limit=2,
        report_to="none", dataloader_num_workers=0, remove_unused_columns=False, dataloader_pin_memory=False
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()

    print(" Saving Builder LoRA...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(" Merging Builder LoRA...")
    gc.collect(); torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda:0")
    peft_model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"✅ BUILDER DONE! Merged saved to {MERGED_DIR}")

if __name__ == "__main__":
    main()
