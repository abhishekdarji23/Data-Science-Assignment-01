"""
Pretrains the nano transformer on data/corpus.txt (next-character prediction).

Usage:
    python train.py
"""

import torch
from model import CharTokenizer, NanoTransformer

BLOCK_SIZE = 64
BATCH_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
LR = 3e-3
STEPS = 800
EVAL_EVERY = 100

torch.manual_seed(1337)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = open("data/corpus.txt", encoding="utf-8").read()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = NanoTransformer(
        vocab_size=tok.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab_size={tok.vocab_size}  params={n_params:,}  device={device}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % EVAL_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, BLOCK_SIZE, BATCH_SIZE, device)
                _, val_loss = model(xv, yv)
            model.train()
            print(f"step {step:4d} | train loss {loss.item():.3f} | val loss {val_loss.item():.3f}")

    torch.save(
        {"model_state": model.state_dict(), "config": model.config, "tokenizer": tok.state_dict()},
        "checkpoints/pretrained.pt",
    )
    print("saved checkpoints/pretrained.pt")

    # quick sanity sample
    model.eval()
    prompt = "The robot"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=120)
    print("\n--- sample ---")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
