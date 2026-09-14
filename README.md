# Language AI Story

A live course notebook on language AI and agentic AI, from a single
neuron up to retrieval-augmented, multi-hop reasoning systems.

**Live, interactive site:** https://karthikv1392.github.io/language-ai-story/
([`index.html`](index.html)) — nineteen hands-on stops in two parts, each
one computed live in the browser as you click, type, and drag. No detailed
reading required — the interactions carry the teaching.

**Part 1 · Language AI** — McCulloch-Pitts neuron, perceptron, multi-layer
perceptron, backpropagation, embeddings, RNNs, encoder-decoder,
self-attention, why Transformers replaced RNNs, token generation,
foundation models, fine-tuning, instruction & chat tuning.

**Part 2 · Agentic AI** — chunking, vector databases, retrieval, RAG,
query rewriting, multi-hop RAG. All six share one small, real knowledge
base (built from Part 1's own history) so retrieval results are honest
and checkable by eye, not just plausible-looking.

## Companion scripts

Four small, heavily-commented Python scripts for walking through attention
and text generation on the command line, in order.

### Setup

```
pip install numpy matplotlib torch
```

(numpy + matplotlib already present; `torch` only needed for file 04.)

### Suggested lecture flow

1. **`01_self_attention_from_scratch.py`**
   Single self-attention head in raw NumPy. Prints every intermediate
   matrix (Q, K, V, scores, softmax weights, output) and saves
   `attention_heatmap.png` — the sentence *"The animal didn't cross the
   street because it was too tired"*, showing "it" attending mostly to
   "animal", not the nearer word "street". This is the one image to put
   on a slide when explaining attention.

2. **`02_multihead_attention.py`**
   Same idea, but 4 heads in parallel, each able to specialise in a
   different relationship (coreference vs. verb-object here). Saves
   `multihead_attention_heatmaps.png` — a side-by-side grid of heads so
   students can see they learn *different* patterns.

3. **`03_how_llms_generate_tokens.py`**
   Wraps causal (masked) self-attention in the full
   embed → attend → logits → softmax → sample → append → repeat loop,
   with **untrained/random weights on purpose** — output is gibberish,
   which is the point: this file is about the *mechanism*, not the
   output quality. Great for explaining "an LLM only ever predicts one
   next token at a time" and what temperature does.

4. **`04_train_tiny_gpt_pytorch.py`**
   The exact same architecture and generation loop, now actually
   trained (PyTorch, ~14k parameters, a few seconds on CPU) on a short
   corpus. Prints generations before training (gibberish, like file 03)
   and after training (fluent, grammatical completions) — the "payoff"
   demo that makes the mechanism-vs-scale distinction click: real LLMs
   are this same loop with ~a billion times more trained parameters.

Each file is standalone — `python3 0N_*.py` — and safe to re-run or
project live; all randomness is seeded for reproducible output.
