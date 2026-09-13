"""
MULTI-HEAD ATTENTION
=====================
Teaching goal: show why real transformers don't use ONE attention
computation, but several in parallel ("heads").

Recap from 01_self_attention_from_scratch.py: one attention head computes
ONE way of relating every token to every other token (one Q/K/V
projection). But language has many kinds of relationships at once, e.g.:
  - "it" -> "animal"     (what does this pronoun refer to?)
  - "cross" -> "street"  (verb <-> its object)
  - "didn't" -> "cross"  (negation attaches to which verb?)

One head has to compress ALL of these into a single similarity pattern.
Multi-head attention instead runs several SMALLER attention heads in
parallel, each with its own Wq/Wk/Wv, so each head is free to specialise
in a different kind of relationship. Their outputs are concatenated and
mixed back together with one more learned matrix, Wo.

This file reuses the same toy sentence and reimplements attention as a
reusable function, then runs it num_heads times with different weights.

Run:  python3 02_multihead_attention.py
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)
rng = np.random.default_rng(seed=7)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(X, Wq, Wk, Wv, boost=None):
    """One scaled dot-product attention head. Returns (output, weights).

    `boost`, if given, is a dict {(query_idx, key_idx): amount} added
    directly to the raw scores before softmax. This is NOT part of real
    attention - it's a teaching shortcut that hand-picks one attention
    pattern per head so the demo reliably shows "different heads learn
    different relationships" without needing actual training.
    """
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    if boost:
        for (qi, ki), amount in boost.items():
            scores[qi, ki] += amount
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


# -----------------------------------------------------------------------
# Toy sentence, same idea as file 01 but we hand-craft TWO separate
# relationships so two different heads can each "specialise" in one:
#   head A will be pushed to link "it" -> "animal"    (coreference)
#   head B will be pushed to link "cross" -> "street"  (verb-object)
# -----------------------------------------------------------------------
tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
n, d_model = len(tokens), 8
num_heads = 4
d_head = d_model // num_heads  # each head works in a smaller subspace

X = rng.normal(size=(n, d_model))

it_idx, animal_idx = tokens.index("it"), tokens.index("animal")
cross_idx, street_idx = tokens.index("cross"), tokens.index("street")

# One independent (Wq, Wk, Wv) triple per head.
heads = [
    (rng.normal(scale=0.5, size=(d_model, d_head)),
     rng.normal(scale=0.5, size=(d_model, d_head)),
     rng.normal(scale=0.5, size=(d_model, d_head)))
    for _ in range(num_heads)
]

# Hand-craft head 0 to specialise in "it" -> "animal", and head 1 to
# specialise in "cross" -> "street" - illustrating that different heads
# can pick up different relationships. (In a trained model this division
# of labour emerges naturally from gradient descent, not hand-crafting -
# see the `boost` docstring in attention() above.)
boosts = {0: {(it_idx, animal_idx): 8.0}, 1: {(cross_idx, street_idx): 8.0}}

head_outputs = []
head_weights = []

for h, (Wq, Wk, Wv) in enumerate(heads):
    out, w = attention(X, Wq, Wk, Wv, boost=boosts.get(h))
    head_outputs.append(out)
    head_weights.append(w)
    print(f"\n--- Head {h} ---")
    print(f"'{tokens[it_idx]}' attends most to:",
          tokens[np.argmax(w[it_idx])], f"({w[it_idx].max():.2f})")
    print(f"'{tokens[cross_idx]}' attends most to:",
          tokens[np.argmax(w[cross_idx])], f"({w[cross_idx].max():.2f})")

# -----------------------------------------------------------------------
# Concatenate all heads' outputs, then project back to d_model with Wo.
# This is the step that lets the model COMBINE what every head found.
# -----------------------------------------------------------------------
concat = np.concatenate(head_outputs, axis=-1)   # shape [n, num_heads * d_head] == [n, d_model]
Wo = rng.normal(scale=0.5, size=(d_model, d_model))
multihead_output = concat @ Wo

print("\nConcatenated heads shape:", concat.shape, " -> after Wo projection:", multihead_output.shape)
print("\nMulti-head attention output (context-aware representations):")
print(multihead_output)

print("""
Takeaway for students:
  Head 0 learned (was hand-crafted) to resolve "it" -> "animal".
  Head 1 learned (was hand-crafted) to resolve "cross" -> "street".
  Heads 2 and 3 are free / random, standing in for the many other
  relationships a real trained model's extra heads would specialise in
  (tense, sentiment, position, syntax, ...).
  Concatenating + projecting (Wo) lets the next layer use ALL of that
  information at once.
""")


def plot_heads(head_weights, tokens, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(head_weights), figsize=(4.2 * len(head_weights), 4.5))
    for h, (ax, w) in enumerate(zip(axes, head_weights)):
        im = ax.imshow(w, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.7, label="attention weight")
    fig.suptitle("Each head learns a different attention pattern")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap grid to {path}")


if __name__ == "__main__":
    plot_heads(head_weights, tokens, "multihead_attention_heatmaps.png")
