"""
Supervised fine-tuning (SFT): takes the pretrained model and continues
training it on a small set of prompt/response pairs, so it learns to
respond to a prompt rather than just continue arbitrary text.

Usage:
    python sft.py
"""

import json
import torch
from model import CharTokenizer, NanoTransformer

LR = 1e-3
EPOCHS = 60
PROMPT_TAG = "\n### Q: "
RESPONSE_TAG = "\n### A: "
END_TAG = "\n### END"


def build_examples(pairs, tok, block_size):
    examples = []
    for pair in pairs:
        full_text = PROMPT_TAG + pair["prompt"] + RESPONSE_TAG + pair["response"] + END_TAG
        ids = tok.encode(full_text)
        if len(ids) < 2:
            continue
        ids = ids[: block_size + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        pad = block_size - len(x)
        if pad > 0:
            x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad,), -1, dtype=torch.long)])
        examples.append((x, y))
    return examples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/pretrained.pt", map_location=device)

    tok = CharTokenizer.from_state_dict(ckpt["tokenizer"])
    cfg = ckpt["config"]
    model = NanoTransformer(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    pairs = json.load(open("data/sft_pairs.json", encoding="utf-8"))
    examples = build_examples(pairs, tok, cfg["block_size"])
    print(f"{len(examples)} SFT examples, block_size={cfg['block_size']}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x, y in examples:
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits, _ = model(x)
            # ignore padded targets (-1) in the loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | avg loss {total_loss / len(examples):.3f}")

    torch.save(
        {"model_state": model.state_dict(), "config": cfg, "tokenizer": ckpt["tokenizer"]},
        "checkpoints/sft.pt",
    )
    print("saved checkpoints/sft.pt")

    # quick sanity check
    model.eval()
    test_prompt = PROMPT_TAG + "What does the robot do in the evening?" + RESPONSE_TAG
    idx = torch.tensor([tok.encode(test_prompt)], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=80, temperature=0.7)
    print("\n--- sample ---")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
