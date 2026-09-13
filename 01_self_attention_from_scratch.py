"""
SELF-ATTENTION FROM SCRATCH
============================
Teaching goal: show students EXACTLY what "attention" computes, using
nothing but NumPy. No frameworks, no magic.

The core idea:
  For every word in a sentence, attention asks "which other words in this
  sentence should I look at, and how much, to understand THIS word better?"

  Example: "The animal didn't cross the street because it was too tired."
  To understand what "it" refers to, a model needs to look back at "animal"
  much more than at "street". Attention is the mechanism that computes
  those "how much to look at each word" weights, automatically, from data.

We will:
  1. Represent a toy sentence as a sequence of word embeddings (vectors).
  2. Turn each embedding into a Query (Q), Key (K), and Value (V) vector.
  3. Score every word against every other word:  score = Q . K
  4. Turn scores into probabilities with softmax  -> the "attention weights"
  5. Use those weights to build a new representation for each word, as a
     weighted mix of everyone's Value vector.
  6. Visualize the attention weights as a heatmap.

Run:  python3 01_self_attention_from_scratch.py
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)
rng = np.random.default_rng(seed=42)


# -----------------------------------------------------------------------
# STEP 0: A toy sentence and toy embeddings
# -----------------------------------------------------------------------
# In a real LLM, each word (technically each "token") is mapped to a vector
# of a few hundred/thousand numbers, learned during training. Here we just
# make up small 4-dimensional vectors so the arithmetic stays readable on
# a projector.

tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
d_model = 4  # embedding dimension - tiny on purpose, so every matrix fits on screen

# Fixed random embeddings (in a real model these come from a learned
# embedding table + are updated during training).
X = rng.normal(size=(len(tokens), d_model))

print("Tokens:", tokens)
print("\nInput embeddings X (one row per token, shape = [seq_len, d_model]):")
print(X)


# -----------------------------------------------------------------------
# STEP 1: Project embeddings into Query, Key, Value spaces
# -----------------------------------------------------------------------
# Q, K, V are just three different learned linear transformations of the
# SAME input embedding. Intuition:
#   Query = "what am I looking for?"      (asked by the current word)
#   Key   = "what do I contain?"          (offered by every word, incl. self)
#   Value = "what information do I give away if someone attends to me?"
#
# In a trained model, Wq/Wk/Wv are learned by gradient descent. Here they
# are random-but-fixed, which is enough to show the MECHANISM.

Wq = rng.normal(scale=0.5, size=(d_model, d_model))
Wk = rng.normal(scale=0.5, size=(d_model, d_model))
Wv = rng.normal(scale=0.5, size=(d_model, d_model))

Q = X @ Wq   # shape [seq_len, d_model]
K = X @ Wk
V = X @ Wv

# With random embeddings + random Wq/Wk, dot products come out roughly
# uniform - no token has a linguistic reason to prefer another yet. A
# TRAINED model learns Wq/Wk so that e.g. the query for "it" ends up
# pointing in the same direction as the key for "animal". We hand-craft
# that single outcome here - nothing else - purely so the printed numbers
# tell the "it" -> "animal" story clearly on a projector.
it_idx, animal_idx = tokens.index("it"), tokens.index("animal")
Q[it_idx] = 2.0 * K[animal_idx] + rng.normal(scale=0.05, size=d_model)

print("\nQ (queries):\n", Q)
print("\nK (keys):\n", K)
print("\nV (values):\n", V)


# -----------------------------------------------------------------------
# STEP 2: Score every query against every key
# -----------------------------------------------------------------------
# raw_scores[i, j] = how much token i should attend to token j
#                  = dot product of query_i and key_j
# Dot product is large & positive when the two vectors point the same way,
# i.e. when "what i is looking for" matches "what j offers".

raw_scores = Q @ K.T   # shape [seq_len, seq_len]

# Scale by sqrt(d_model). Why? Dot products grow with vector length, which
# would push softmax into a saturated, near-one-hot regime and make
# training unstable for large d_model. Dividing keeps the scale sane.
scaled_scores = raw_scores / np.sqrt(d_model)

print("\nRaw attention scores (Q @ K^T), shape [seq_len, seq_len]:")
print(raw_scores)
print("\nScaled attention scores (divided by sqrt(d_model)):")
print(scaled_scores)


# -----------------------------------------------------------------------
# STEP 3: Softmax each row -> attention weights that sum to 1
# -----------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


attention_weights = softmax(scaled_scores, axis=-1)

print("\nAttention weights (each row sums to 1 - 'how much token i looks at every token j'):")
print(attention_weights)

print(f"\nZoom in on the row for the word '{tokens[it_idx]}':")
for tok, w in sorted(zip(tokens, attention_weights[it_idx]), key=lambda p: -p[1]):
    print(f"   {tok:10s} {w:.3f}")
print("-> notice 'it' pays most attention to 'animal', not to 'street',")
print("   even though 'street' is closer in the sentence. That's what")
print("   makes attention more powerful than just looking at nearby words.")


# -----------------------------------------------------------------------
# STEP 4: Use the weights to mix Value vectors -> new representation
# -----------------------------------------------------------------------
# output[i] = weighted average of ALL value vectors, weighted by how much
# token i attends to each token. This is the token's new, "context-aware"
# representation - e.g. the vector for "it" now has "animal" mixed into it.

output = attention_weights @ V   # shape [seq_len, d_model]

print("\nOutput of self-attention (context-aware representations):")
print(output)
print("\n(In a real transformer this output is added back to the input")
print(" (a 'residual connection'), passed through a feed-forward layer,")
print(" and this whole block is stacked dozens of times.)")


# -----------------------------------------------------------------------
# STEP 5: Visualize the attention weights as a heatmap
# -----------------------------------------------------------------------
def plot_attention(weights, tokens, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("attending TO (Key)")
    ax.set_ylabel("attending FROM (Query)")
    ax.set_title("Self-Attention Weights")
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                     color="white" if weights[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved heatmap to {path}")


if __name__ == "__main__":
    plot_attention(attention_weights, tokens, "attention_heatmap.png")
