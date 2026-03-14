"""
Evaluation for trained Word2Vec embeddings.

Two evaluation methods:

1. WORD ANALOGY (3CosAdd)
   The classic Mikolov et al. test.  Given three words a, b, c, find the
   word w* that best completes the analogy "a is to b as c is to ?":

       w* = argmax  cos( w,  v_b − v_a + v_c )

   We test against the Google analogy dataset (~19,500 questions split into
   semantic categories like capital cities, currencies, and syntactic
   categories like plural nouns, verb tenses).

   The file is downloaded automatically on first run.

2. NEAREST NEIGHBOURS
   Cosine-similarity lookup: find the top-k words closest to a query word.
   Useful as a quick sanity check after training.

Run:
    python word2vec/evaluate.py
"""

import os
import sys
import json
import urllib.request
import numpy as np
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_PATH  = os.path.join(_HERE, "embeddings.npy")
VOCAB_PATH       = os.path.join(_HERE, "vocab.json")
ANALOGIES_URL    = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
ANALOGIES_PATH   = os.path.join(_HERE, "questions-words.txt")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(embeddings_path=EMBEDDINGS_PATH, vocab_path=VOCAB_PATH):
    """Load embeddings and vocabulary from disk.

    Returns:
        W         (np.ndarray, V x d): L2-normalised embedding matrix.
        word_to_id (dict):             word -> integer id.
        id_to_word (dict):             integer id -> word.
    """
    W = np.load(embeddings_path)

    with open(vocab_path, "r") as f:
        word_to_id = json.load(f)
    id_to_word = {i: w for w, i in word_to_id.items()}

    # L2-normalise once so all similarity queries are just dot products
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_n   = W / norms

    print(f"Loaded embeddings: {W.shape}  vocab: {len(word_to_id):,}")
    return W_n, word_to_id, id_to_word


# ---------------------------------------------------------------------------
# Nearest neighbours
# ---------------------------------------------------------------------------

def nearest_neighbours(W_n, word_to_id, id_to_word, query, top_k=10):
    """Print the top_k most similar words to `query` by cosine similarity.

    Because W_n is already L2-normalised, cosine similarity reduces to a
    plain dot product:  cos(u, v) = u · v  (when ||u|| = ||v|| = 1).

    Args:
        W_n:       L2-normalised embedding matrix, shape (V, d).
        word_to_id: word -> id mapping.
        id_to_word: id -> word mapping.
        query:     word string to look up.
        top_k:     number of neighbours to return.
    """
    if query not in word_to_id:
        print(f"  '{query}' not in vocabulary.")
        return

    q_vec = W_n[word_to_id[query]]          # (d,)
    sims  = W_n @ q_vec                     # (V,) cosine similarities
    sims[word_to_id[query]] = -1.0          # exclude the query word itself

    top_ids = np.argsort(-sims)[:top_k]
    print(f"\nNearest neighbours to '{query}':")
    for rank, idx in enumerate(top_ids, 1):
        print(f"  {rank:2d}. {id_to_word[idx]:<20s}  cosine={sims[idx]:.4f}")


# ---------------------------------------------------------------------------
# Word analogy evaluation (3CosAdd)
# ---------------------------------------------------------------------------

def _download_analogies():
    """Download the Google analogy file if not already present."""
    if not os.path.exists(ANALOGIES_PATH):
        print(f"Downloading analogy file from {ANALOGIES_URL} ...")
        urllib.request.urlretrieve(ANALOGIES_URL, ANALOGIES_PATH)
        print("Download complete.")


def _parse_analogies(path):
    """Parse the questions-words.txt file.

    The file has section headers starting with ':' and then lines of four words:
        a  b  c  d
    representing the analogy "a is to b as c is to d".

    Returns:
        List of (category_name, a, b, c, d) tuples, all lowercase.
    """
    questions = []
    category  = "unknown"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            if line.startswith(":"):
                category = line[2:]   # e.g. "capital-world"
            else:
                parts = line.split()
                if len(parts) == 4:
                    questions.append((category, *parts))
    return questions


def evaluate_analogies(W_n, word_to_id, path=ANALOGIES_PATH, top_k=1):
    """Evaluate embeddings on the Google word analogy benchmark.

    Method — 3CosAdd (Mikolov et al. 2013):
        Given words a, b, c, predict d such that "a:b :: c:d"
        by finding:

            w* = argmax  cos( w,  v_b − v_a + v_c )

        with the constraint that w ∉ {a, b, c} (exclude input words).

    The query vector v_b - v_a + v_c can be thought of as:
        "start at c, then apply the same offset that separates a from b."
    For "man:king :: woman:?", the offset king-man encodes "royalty",
    and woman + royalty ≈ queen.

    The matrix form makes this fast: after computing the query vector q,
    cosine similarities to all V words are just W_n @ q (one matrix-vector
    multiply), so the full benchmark runs in seconds.

    Args:
        W_n:       L2-normalised embeddings, shape (V, d).
        word_to_id: word -> id.
        path:      path to questions-words.txt.
        top_k:     prediction is correct if answer is in top k results.

    Returns:
        overall accuracy (float).
    """
    _download_analogies()
    questions = _parse_analogies(path)

    # Group by category for per-category reporting
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)
    skipped     = 0

    for category, a, b, c, d in questions:
        # Skip if any word is outside vocabulary
        if any(w not in word_to_id for w in (a, b, c, d)):
            skipped += 1
            continue

        ia, ib, ic, id_ = (word_to_id[w] for w in (a, b, c, d))

        # Query vector: v_b - v_a + v_c  (all already L2-normalised)
        query = W_n[ib] - W_n[ia] + W_n[ic]     # (d,)
        # No need to renormalise query — we just need the argmax

        # Cosine similarities to all words
        sims = W_n @ query                        # (V,)

        # Mask out the three input words (they shouldn't be the answer)
        sims[ia] = -np.inf
        sims[ib] = -np.inf
        sims[ic] = -np.inf

        top_ids = np.argsort(-sims)[:top_k]
        cat_total[category]   += 1
        if id_ in top_ids:
            cat_correct[category] += 1

    # Print per-category results
    print(f"\n{'Category':<35s}  {'Correct':>7}  {'Total':>7}  {'Acc':>6}")
    print("-" * 62)

    all_correct = 0
    all_total   = 0
    for cat in sorted(cat_total):
        correct = cat_correct[cat]
        total   = cat_total[cat]
        acc     = correct / total if total > 0 else 0.0
        print(f"  {cat:<33s}  {correct:>7,}  {total:>7,}  {acc:>5.1%}")
        all_correct += correct
        all_total   += total

    overall = all_correct / all_total if all_total > 0 else 0.0
    print("-" * 62)
    print(f"  {'OVERALL':<33s}  {all_correct:>7,}  {all_total:>7,}  {overall:>5.1%}")
    if skipped:
        print(f"\n  ({skipped:,} questions skipped — words not in vocabulary)")

    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embeddings_path = sys.argv[1] if len(sys.argv) > 1 else EMBEDDINGS_PATH
    W_n, word_to_id, id_to_word = load(embeddings_path)

    # Nearest-neighbour sanity check
    print("\n" + "=" * 62)
    print("Nearest-neighbour examples")
    print("=" * 62)
    for word in ["king", "france", "football", "computer"]:
        nearest_neighbours(W_n, word_to_id, id_to_word, word, top_k=8)

    # Analogy benchmark
    print("\n" + "=" * 62)
    print("Google analogy benchmark (3CosAdd)")
    print("=" * 62)
    evaluate_analogies(W_n, word_to_id)
