"""
A TINY GPT THAT ACTUALLY LEARNS (PyTorch)
===========================================
Teaching goal: file 03 showed the generation MECHANISM with random,
untrained weights (so the output was gibberish by design). This file is
the payoff - the exact same architecture (embeddings -> causal self-
attention -> unembedding -> softmax -> sample -> append -> repeat),
except now we actually TRAIN it with gradient descent on a short corpus,
so students can watch it go from random noise to real completions in a
few seconds.

This is intentionally the smallest possible "GPT": one attention head,
one transformer block, a few thousand parameters, trained on one short
paragraph. Real LLMs are the same architecture (see attention() below -
it is line-for-line the same idea as 01_self_attention_from_scratch.py)
scaled up to ~100 layers, ~100 attention heads per layer, and trained on
trillions of tokens instead of ~200 characters.

Run:  python3 04_train_tiny_gpt_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# -----------------------------------------------------------------------
# STEP 1: Tokenize (character-level, same idea as file 03)
# -----------------------------------------------------------------------
corpus = (
    "the cat sat on the mat. "
    "the dog sat on the rug. "
    "the cat chased the dog. "
    "the dog chased the cat. "
)
vocab = sorted(set(corpus))
vocab_size = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(corpus), dtype=torch.long)
print(f"Corpus: {corpus!r}")
print(f"Vocabulary ({vocab_size} tokens): {vocab}")

block_size = 16   # how many previous tokens the model can look at
device = "cpu"


def get_batch(batch_size=32):
    """Sample random (context, target) windows for training.
    target[t] is always the character that comes right after context[:t+1]
    - this is exactly the "predict the next token" objective LLMs train on."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# -----------------------------------------------------------------------
# STEP 2: The model - embeddings + ONE causal self-attention block
# -----------------------------------------------------------------------
d_model = 32
n_head = 4


class CausalSelfAttention(nn.Module):
    """This is literally the same computation as 01_self_attention_from_scratch.py
    (Q, K, V projections -> scaled dot-product scores -> softmax -> weighted
    sum of V), plus multiple heads (like file 02) and a causal mask (like
    file 03), except now Wq/Wk/Wv/Wo are nn.Linear layers whose weights get
    UPDATED by backpropagation instead of being fixed/random."""

    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        causal_mask = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T, C = x.shape  # batch, time (sequence length), channels (d_model)

        def split_heads(t):
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, heads, T, d_head]

        Q, K, V = split_heads(self.Wq(x)), split_heads(self.Wk(x)), split_heads(self.Wv(x))

        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)   # [B, heads, T, T]
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                            # [B, heads, T, d_head]

        out = out.transpose(1, 2).contiguous().view(B, T, C)          # concat heads
        return self.Wo(out)


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffwd = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size)   # maps back to vocabulary logits
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = x + self.attn(self.ln1(x))     # residual connection around attention
        x = x + self.ffwd(self.ln2(x))     # residual connection around feed-forward
        logits = self.unembed(x)            # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, num_new_tokens, temperature=1.0, verbose=False):
        """The exact same autoregressive loop as file 03's generate():
        predict next-token logits -> softmax -> sample -> append -> repeat."""
        for _ in range(num_new_tokens):
            context = idx[:, -self.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature   # only last position matters
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if verbose:
                top = torch.topk(probs[0], k=3)
                choices = ", ".join(f"{itos[i.item()]!r}:{p.item():.2f}" for p, i in zip(top.values, top.indices))
                print(f"  context={decode(idx[0].tolist())[-20:]!r:22s} top -> {choices}")
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = TinyGPT(vocab_size, d_model, n_head, block_size).to(device)
print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters "
      f"(GPT-3 has ~175,000,000,000 - about a billion times more).")


# -----------------------------------------------------------------------
# STEP 3: BEFORE training - show it's just as gibberish as file 03
# -----------------------------------------------------------------------
prompt = "the cat "
prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long)

print("\n=== BEFORE training (random weights) ===")
out = model.generate(prompt_ids.clone(), num_new_tokens=40)
print("Generated:", decode(out[0].tolist()))


# -----------------------------------------------------------------------
# STEP 4: Train with plain gradient descent on the next-token prediction task
# -----------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

print("\n=== Training (next-token prediction / cross-entropy loss) ===")
for step in range(400):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == 399:
        print(f"step {step:4d} | loss {loss.item():.3f}")


# -----------------------------------------------------------------------
# STEP 5: AFTER training - same exact generation loop, trained weights
# -----------------------------------------------------------------------
print("\n=== AFTER training ===")
out = model.generate(prompt_ids.clone(), num_new_tokens=60, verbose=True)
print("\nGenerated:", decode(out[0].tolist()))

print("""
Takeaway for students:
  - Same architecture, same generation loop as file 03 - the ONLY thing
    that changed is that Wq/Wk/Wv/Wo/embeddings were adjusted by gradient
    descent to minimise "how surprised was the model by the real next
    character". That single objective (next-token prediction) is what
    modern LLMs like GPT-4/Claude are trained on, just at a vastly
    bigger scale (more data, more parameters, more compute).
  - Lower temperature below (try temperature=0.3) makes it pick the most
    likely token more often -> more repetitive/predictable text.
  - Higher temperature (try temperature=1.5) makes sampling more random
    -> more 'creative' but also more error-prone text.
""")
