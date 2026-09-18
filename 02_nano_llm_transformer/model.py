"""
A tiny, from-scratch decoder-only transformer (a "nanoLLM").

Deliberately small and simple: character-level tokenizer, a couple of
transformer blocks, RoPE (rotary position embeddings) instead of learned
positional embeddings, and a SwiGLU feed-forward network instead of a plain
MLP. It's a teaching-scale model, not a production one — the point is to
see the architecture work end to end (train -> fine-tune -> serve).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer: plain character-level, vocab built from whatever text you give it
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def state_dict(self):
        return {"stoi": self.stoi, "itos": {int(k): v for k, v in self.itos.items()}}

    @classmethod
    def from_state_dict(cls, state):
        tok = cls.__new__(cls)
        tok.stoi = state["stoi"]
        tok.itos = {int(k): v for k, v in state["itos"].items()}
        tok.vocab_size = len(tok.stoi)
        return tok


# ---------------------------------------------------------------------------
# RoPE: rotate query/key pairs by a position-dependent angle instead of
# adding a learned positional embedding to the input.
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len, head_dim, device):
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)  # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (batch, heads, seq_len, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.stack([rotated_even, rotated_odd], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    """FFN(x) = W2( SiLU(W1 x) * W3 x ) — a gated feed-forward network."""

    def __init__(self, n_embd, hidden_mult=4):
        super().__init__()
        hidden = int(n_embd * hidden_mult * 2 / 3)  # keep param count close to a 4x plain MLP
        self.w1 = nn.Linear(n_embd, hidden)
        self.w3 = nn.Linear(n_embd, hidden)
        self.w2 = nn.Linear(hidden, n_embd)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = SwiGLU(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class NanoTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        cos, sin = build_rope_cache(block_size, self.head_dim, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.config = dict(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)

        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
