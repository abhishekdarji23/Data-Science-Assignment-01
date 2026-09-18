"""
Minimal FastAPI server that loads the trained nano transformer and exposes
one endpoint for generating text from a prompt.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8002
"""

import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import CharTokenizer, NanoTransformer
from sft import PROMPT_TAG, RESPONSE_TAG, END_TAG

CKPT_PATH = "checkpoints/sft.pt" if os.path.exists("checkpoints/sft.pt") else "checkpoints/pretrained.pt"

app = FastAPI(title="Nano Transformer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT_PATH, map_location=device)
tokenizer = CharTokenizer.from_state_dict(ckpt["tokenizer"])
model = NanoTransformer(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"loaded {CKPT_PATH} | vocab_size={tokenizer.vocab_size} | device={device}")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    mode: str = "chat"  # "chat" wraps the prompt in the SFT Q/A format, "raw" doesn't


class GenerateResponse(BaseModel):
    text: str
    checkpoint: str


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CKPT_PATH, "vocab_size": tokenizer.vocab_size, "device": device}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.mode == "chat":
        text = PROMPT_TAG + req.prompt + RESPONSE_TAG
    else:
        text = req.prompt

    ids = tokenizer.encode(text)
    if not ids:
        return GenerateResponse(text="(prompt has no known characters)", checkpoint=CKPT_PATH)

    idx = torch.tensor([ids], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    decoded = tokenizer.decode(out[0].tolist())

    completion = decoded[len(text):]
    if req.mode == "chat" and END_TAG in completion:
        completion = completion.split(END_TAG)[0]

    return GenerateResponse(text=completion.strip(), checkpoint=CKPT_PATH)
