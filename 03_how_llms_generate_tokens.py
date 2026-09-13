"""
HOW LLMs GENERATE TOKENS (the mechanism, in NumPy)
====================================================
Teaching goal: show the actual LOOP an LLM runs to produce text, one
token at a time. This file builds a tiny transformer decoder (embeddings
+ causal self-attention + a linear "unembedding" layer to vocabulary
logits) with NumPy only.

IMPORTANT for students: the weights below are randomly initialised, NOT
trained. So the generated text will be gibberish. That is intentional -
this file is not about making a good model, it's about making the
GENERATION MECHANISM visible:

    tokenize -> embed -> (causal self-attention, many layers in a real
    model) -> logits over the whole vocabulary -> softmax -> pick a
    token -> APPEND it to the input -> repeat

A real LLM (GPT-4, Claude, Llama, ...) runs exactly this loop. The only
differences are scale (billions of trained parameters instead of a few
hundred random ones) and that every weight has been trained on trillions
of tokens of text so the softmax distribution is actually meaningful.
See 04_train_tiny_gpt_pytorch.py for a version that is actually trained.

Run:  python3 03_how_llms_generate_tokens.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(seed=0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# -----------------------------------------------------------------------
# STEP 1: Tokenization
# -----------------------------------------------------------------------
# Real LLMs use subword tokenizers (BPE etc.) that split text into ~30k-
# 100k possible chunks. Here we use CHARACTERS as tokens - same idea,
# much smaller vocabulary, so it's easy to inspect on screen.

corpus = "the cat sat on the mat"
vocab = sorted(set(corpus))
vocab_size = len(vocab)
char_to_id = {c: i for i, c in enumerate(vocab)}
id_to_char = {i: c for i, c in enumerate(vocab)}

print("Vocabulary (all tokens the model can possibly produce):")
print(vocab, f"  ({vocab_size} tokens)")


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(id_to_char[i] for i in ids)


# -----------------------------------------------------------------------
# STEP 2: Model "parameters" - randomly initialised (NOT trained)
# -----------------------------------------------------------------------
d_model = 16
block_size = 32  # max context length this toy model supports

E = rng.normal(scale=0.1, size=(vocab_size, d_model))   # token embedding table
P = rng.normal(scale=0.1, size=(block_size, d_model))   # positional embeddings
Wq = rng.normal(scale=0.1, size=(d_model, d_model))
Wk = rng.normal(scale=0.1, size=(d_model, d_model))
Wv = rng.normal(scale=0.1, size=(d_model, d_model))
Wo = rng.normal(scale=0.1, size=(d_model, d_model))
Wunembed = rng.normal(scale=0.1, size=(d_model, vocab_size))  # maps back to vocab logits


def causal_self_attention(X):
    """Same attention as file 01, but with a CAUSAL MASK: token i may only
    attend to tokens <= i. This is what makes generation well-defined -
    a model must not be allowed to "peek" at tokens it hasn't produced yet."""
    n = X.shape[0]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = (Q @ K.T) / np.sqrt(d_model)

    # Causal mask: set scores for "future" positions to -infinity so
    # softmax gives them ~0 probability.
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    scores[mask] = -np.inf

    weights = softmax(scores, axis=-1)
    return (weights @ V) @ Wo, weights


def forward(token_ids):
    """One forward pass: token ids -> logits over the vocabulary for
    what comes NEXT after each position."""
    n = len(token_ids)
    X = E[token_ids] + P[:n]           # embed tokens + add position info
    attn_out, attn_weights = causal_self_attention(X)
    X = X + attn_out                    # residual connection
    logits = X @ Wunembed               # shape [n, vocab_size]
    return logits, attn_weights


# -----------------------------------------------------------------------
# STEP 3: The autoregressive generation loop
# -----------------------------------------------------------------------
# This is the heart of "how an LLM produces text". At every step:
#   1. Run the WHOLE sequence so far through the model.
#   2. Look only at the logits for the LAST position (what comes next).
#   3. Softmax -> a probability distribution over every possible token.
#   4. Pick one token (here: sample from that distribution).
#   5. Append it to the sequence and repeat.

def generate(prompt, num_new_tokens=15, temperature=1.0, seed=1):
    gen_rng = np.random.default_rng(seed)
    ids = encode(prompt)
    print(f"\nPrompt: {prompt!r}  ->  token ids: {ids}")

    for step in range(num_new_tokens):
        context = ids[-block_size:]
        logits, _ = forward(context)
        next_logits = logits[-1] / temperature      # only the LAST position matters
        probs = softmax(next_logits)

        top5 = np.argsort(probs)[::-1][:5]
        top5_str = ", ".join(f"{id_to_char[t]!r}:{probs[t]:.2f}" for t in top5)
        print(f"step {step:2d} | context={decode(context)!r:26s} | top choices -> {top5_str}")

        next_id = gen_rng.choice(vocab_size, p=probs)  # SAMPLE the next token
        ids.append(next_id)

    print(f"\nFinal generated sequence: {decode(ids)!r}")
    return ids


if __name__ == "__main__":
    generate("the cat ", num_new_tokens=15)

    print("""
Takeaway for students:
  - The model NEVER "decides on a whole sentence" - it only ever
    predicts a probability distribution for ONE next token at a time.
  - Every step re-reads the ENTIRE sequence so far (that's what the
    causal self-attention above is doing) to decide that distribution.
  - "temperature" controls how random the choice is: temperature -> 0
    means always pick the single most likely token (deterministic,
    'greedy' decoding); higher temperature flattens the distribution
    and makes output more random/creative.
  - Because these weights are untrained, the distribution is close to
    uniform and the output is gibberish. Training (see file 04) is what
    shapes this distribution so the sampled tokens form real language.
""")
