import os, sys, torch, time, uuid, traceback, warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── ROCm & HF Env (Wajib) ────────────────────────────────────────────────
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"   # ← FIX utama
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

app = FastAPI(docs_url=None, redoc_url=None)
model = None
tokenizer = None
MODEL_NAME = "local"
DEVICE = "cuda:0"
LOG_FILE = None

class ChatMessage(BaseModel): role: Literal["system", "user", "assistant"]; content: str
class ChatRequest(BaseModel): model: Optional[str] = None; messages: List[ChatMessage]; temperature: Optional[float] = 0.7; max_tokens: Optional[int] = 2048; response_format: Optional[dict] = None
class ChatResponse(BaseModel): id: str; object: str = "chat.completion"; created: int; model: str; choices: List[dict]; usage: dict

def setup_logging(log_path):
    global LOG_FILE
    LOG_FILE = open(log_path, "a")
    sys.stderr = LOG_FILE
    sys.stdout = LOG_FILE

def load_model(path: str):
    global model, tokenizer, MODEL_NAME
    print(f"📦 Loading {path}...")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 🔒 FORCE LOCAL + STANDARD LOADER (bypasses mmap hang)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        local_files_only=True,   #  Kunci: skip network checksum → no hang
        trust_remote_code=True
    )
    model.eval()
    MODEL_NAME = os.path.basename(path)
    torch.cuda.empty_cache()
    print(f"✅ Loaded {MODEL_NAME} | VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    sys.stdout.flush()

@app.get("/health")
def health(): return {"status": "healthy", "model": MODEL_NAME}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if model is None: raise HTTPException(503, "Model not loaded")
    messages = [m.model_dump() for m in req.messages]
    if req.response_format and req.response_format.get("type") == "json_object":
        j = "\n\n⚠️ Output ONLY valid JSON. No markdown."
        for m in messages:
            if m["role"] == "system": m["content"] += j; break
        else: messages.insert(0, {"role": "system", "content": "JSON-only." + j})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_tokens or 2048, temperature=req.temperature or 0.7, do_sample=(req.temperature or 0.7) > 0, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if req.response_format and req.response_format.get("type") == "json_object":
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return ChatResponse(id=f"chat-{uuid.uuid4().hex[:8]}", created=int(time.time()), model=MODEL_NAME, choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], usage={"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(len(out[0])-inputs["input_ids"].shape[1]), "total_tokens": int(len(out[0]))})

if __name__ == "__main__":
    if len(sys.argv) < 3: print("Usage: python server.py <model_path> <port>"); sys.exit(1)
    
    model_path, port = sys.argv[1], int(sys.argv[2])
    log_name = f"/home/security-ft/code-security-ft/logs/server_{os.path.basename(model_path)}.log"
    setup_logging(log_name)
    
    try:
        load_model(model_path)
        import uvicorn
        print(f"🚀 Starting server on port {port}...")
        sys.stdout.flush()
        uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
    except Exception as e:
        print(f"💥 FATAL CRASH: {e}")
        traceback.print_exc()
        sys.exit(1)
