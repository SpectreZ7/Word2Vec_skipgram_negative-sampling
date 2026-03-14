"""
Training script: Word2Vec skip-gram with negative sampling on text8.

Run from the project root:
    python word2vec/train.py

Or from inside the word2vec/ directory:
    python train.py

GPU support
-----------
Set USE_GPU = True (default) to train on your NVIDIA GPU via CuPy.
CuPy must be installed first — see installation instructions below.

Install CuPy:
    # Check your CUDA version:  nvidia-smi  (look for "CUDA Version: 12.x")
    pip install cupy-cuda12x   # CUDA 12.x  (RTX 4060 Ti with recent drivers)
    pip install cupy-cuda11x   # CUDA 11.x  (older drivers)

Why batching is required for GPU speedup
-----------------------------------------
Launching a GPU kernel per training pair would be SLOWER than CPU — the
kernel-launch overhead (~5-10 µs) dwarfs the actual compute for a single
100-d vector operation. BATCH_SIZE controls how many (center, context) pairs
are processed per kernel call, amortising that overhead across B pairs.
Rule of thumb: larger batches use the GPU more efficiently but introduce
slightly more noise into the gradient estimates.

Quick test:
    Set MAX_TOKENS = 1_000_000 for a fast smoke-test (~1 min on GPU).
    Full text8 training: ~10-20 min on RTX 4060 Ti vs ~1-2 h on CPU.
"""

import sys
import os
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from preprocess import preprocess, subsample_tokens, build_neg_sampler
from model import Word2Vec

# ── Config ────────────────────────────────────────────────────────────────────

USE_GPU    = True        # use CuPy/GPU if available; falls back to CPU automatically
BATCH_SIZE = 512         # (center, context) pairs per GPU kernel call

NUM_EPOCHS    = 3        # number of passes over the corpus
EMBED_DIM     = 100      # dimensionality of word vectors
MAX_WINDOW    = 5        # maximum context-window radius
NUM_NEG       = 5        # negative samples per (center, context) pair
LEARNING_RATE = 0.025    # initial SGD learning rate
MIN_LR        = 1e-4     # learning rate floor (linear decay stops here)
MIN_COUNT     = 5        # discard words with fewer occurrences than this
SUBSAMPLE_T   = 1e-5     # subsampling threshold t
MAX_TOKENS    = None     # set to e.g. 1_000_000 for a quick smoke-test
LOG_EVERY     = 500_000  # print a progress line every this many update steps

DATA_PATH = os.path.join(_HERE, "dataset", "text8.txt")

# ── Device selection ─────────────────────────────────────────────────────────

if USE_GPU:
    try:
        import cupy as xp
        xp.array([1.0])  # smoke-test: triggers CUDA init; catches missing DLLs
        print(f"Device: GPU  ({xp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
    except (ImportError, Exception) as e:
        import numpy as xp
        print(f"Device: CPU  (GPU not available: {e.__class__.__name__}, falling back to NumPy)")
else:
    xp = np
    print("Device: CPU  (USE_GPU=False)")

# ─────────────────────────────────────────────────────────────────────────────


def nearest_neighbours(W_in, word_to_id, id_to_word, query, top_k=10):
    """Print the top_k most similar words to `query` by cosine similarity."""
    if query not in word_to_id:
        print(f"  '{query}' not in vocabulary.")
        return

    # Bring to CPU for display (no-op if already numpy)
    W = W_in.get() if hasattr(W_in, 'get') else W_in

    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_n   = W / norms
    q_vec = W_n[word_to_id[query]]
    sims  = W_n @ q_vec
    sims[word_to_id[query]] = -1.0

    top_ids = np.argsort(-sims)[:top_k]
    print(f"\n  Nearest neighbours to '{query}':")
    for rank, idx in enumerate(top_ids, 1):
        print(f"    {rank:2d}. {id_to_word[idx]:<20s}  cosine={sims[idx]:.4f}")


def _flush_batch(model, batch_centers, batch_contexts, batch_negs, lr):
    """Convert accumulated Python lists to device arrays and run batched update."""
    c    = xp.array(batch_centers,  dtype=xp.int32)
    o    = xp.array(batch_contexts, dtype=xp.int32)
    n    = xp.array(batch_negs,     dtype=xp.int32)   # (B, K)
    loss = model.backward_and_update_batch(c, o, n, lr)
    batch_centers.clear()
    batch_contexts.clear()
    batch_negs.clear()
    return loss


def train():
    rng = np.random.default_rng(42)

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1/3: Preprocessing")
    print("=" * 60)
    token_ids, word_to_id, id_to_word, word_freqs = preprocess(
        DATA_PATH, min_count=MIN_COUNT
    )

    if MAX_TOKENS is not None:
        token_ids = token_ids[:MAX_TOKENS]
        print(f"  Truncated to {MAX_TOKENS:,} tokens (MAX_TOKENS)")

    token_ids = subsample_tokens(token_ids, word_freqs, t=SUBSAMPLE_T, rng=rng)

    # ── 2. Build model + negative sampler ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2/3: Initialising model")
    print("=" * 60)
    V          = len(word_to_id)
    model      = Word2Vec(V, embed_dim=EMBED_DIM, xp=xp)
    neg_sample = build_neg_sampler(word_freqs, rng=rng)   # stays on CPU (fast enough)

    print(f"  Vocabulary  : {V:,}")
    print(f"  W_in shape  : {model.W_in.shape}   (input / center embeddings)")
    print(f"  W_out shape : {model.W_out.shape}   (output / context embeddings)")
    print(f"  Parameters  : {2 * V * EMBED_DIM:,}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Epochs      : {NUM_EPOCHS}")

    # ── 3. Training loop (multi-epoch) ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3/3: Training")
    print("=" * 60)

    N           = len(token_ids)
    total_steps = N * NUM_EPOCHS   # LR decays over the full run, not per epoch

    batch_centers  = []
    batch_contexts = []
    batch_negs     = []

    global_step = 0
    total_loss  = 0.0
    t0          = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{NUM_EPOCHS} ──────────────────────────────────")

        for center_pos in range(N):
            center_id = int(token_ids[center_pos])

            # Sample window radius from [1, MAX_WINDOW]
            w     = int(rng.integers(1, MAX_WINDOW + 1))
            left  = max(0, center_pos - w)
            right = min(N, center_pos + w + 1)

            # LR decays linearly over the entire run (all epochs combined)
            progress = ((epoch - 1) * N + center_pos) / total_steps
            lr = max(LEARNING_RATE * (1.0 - progress), MIN_LR)

            for ctx_pos in range(left, right):
                if ctx_pos == center_pos:
                    continue

                context_id = int(token_ids[ctx_pos])
                neg_ids    = neg_sample(NUM_NEG).tolist()

                batch_centers.append(center_id)
                batch_contexts.append(context_id)
                batch_negs.append(neg_ids)
                global_step += 1

                # When batch is full, dispatch to GPU
                if len(batch_centers) >= BATCH_SIZE:
                    loss        = _flush_batch(model, batch_centers, batch_contexts, batch_negs, lr)
                    total_loss += loss * BATCH_SIZE

                if global_step % LOG_EVERY == 0:
                    elapsed    = time.time() - t0
                    avg_loss   = total_loss / LOG_EVERY
                    total_loss = 0.0
                    pct        = 100.0 * center_pos / N
                    print(
                        f"  [ep{epoch} {pct:5.1f}%] step={global_step:>11,}  "
                        f"loss={avg_loss:.4f}  lr={lr:.6f}  elapsed={elapsed:.0f}s"
                    )
                    t0 = time.time()

        # Flush any remaining pairs at end of epoch
        if batch_centers:
            loss        = _flush_batch(model, batch_centers, batch_contexts, batch_negs, lr)
            total_loss += loss * len(batch_centers)

        # Save + nearest-neighbour check after each epoch
        save_path = os.path.join(_HERE, f"embeddings_epoch{epoch}.npy")
        model.save(save_path)
        for probe in ["king", "france", "football", "computer"]:
            nearest_neighbours(model.W_in, word_to_id, id_to_word, probe)

    print(f"\nTraining complete.  Total steps: {global_step:,}")
    return model, word_to_id, id_to_word


if __name__ == "__main__":
    train()
