# 02 — Nano Transformer

A tiny, from-scratch decoder-only transformer, kept deliberately simple:
plain Python/PyTorch backend, plain HTML/JS frontend, no build step, no
extra frameworks.

## What's inside

- **`model.py`** — the model itself: character-level tokenizer, causal
  self-attention with **RoPE** (rotary position embeddings) instead of
  learned positional embeddings, and a **SwiGLU** feed-forward network
  instead of a plain MLP. ~100k parameters, 2 layers.
- **`train.py`** — pretrains the model on `data/corpus.txt` (next-character
  prediction).
- **`sft.py`** — supervised fine-tuning: takes the pretrained model and
  continues training it on `data/sft_pairs.json` (prompt/response pairs),
  so it learns to answer a question instead of just continuing text.
- **`server.py`** — a small FastAPI app with one real endpoint
  (`POST /generate`) that loads the trained checkpoint and generates text.
- **`frontend/`** — one HTML file and one JS file. No React, no Vite,
  no build step — just `fetch()` calling the backend.

Trained checkpoints (`checkpoints/pretrained.pt`, `checkpoints/sft.pt`) are
included, so the server runs immediately without training first.

## Running it

```bash
pip install -r requirements.txt

# start the backend (port 8002)
uvicorn server:app --port 8002

# in another terminal, serve the frontend (any static file server works)
cd frontend
python3 -m http.server 5175
# open http://localhost:5175
```

Type a prompt and click Generate. Try one of the questions from
`data/sft_pairs.json` (e.g. "What does the robot do in the evening?") to see
the fine-tuned behavior, or a free-form line to see it fall back to
continuing text in the style of the training corpus.

## Retraining from scratch

```bash
python train.py   # pretrain on data/corpus.txt -> checkpoints/pretrained.pt
python sft.py      # fine-tune on data/sft_pairs.json -> checkpoints/sft.pt
```

## Being honest about scale

This is a "nano" model on purpose — a few hundred lines of text, ~100k
parameters, trained in seconds on a CPU. It will mostly reproduce patterns
it memorized from its tiny training set rather than generalize like a real
LLM. The point of this project is to see every piece of the architecture
(RoPE, SwiGLU, causal attention, pretrain → SFT → serve) working correctly
end to end, not to build something fluent.
