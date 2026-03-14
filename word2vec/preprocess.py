
from pathlib import Path
import json
import os
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Low-level helpers (unchanged from original)
# ---------------------------------------------------------------------------

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def tokenise(text):
    return text.split()


def get_word_count(tokens):
    return Counter(tokens)


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(file_path='word2vec/dataset/text8.txt', min_count=5):
    """Load text8, build vocabulary, and return integer token ids.

    Steps:
      1. Load raw text and split on whitespace.
      2. Count all word frequencies.
      3. Discard words appearing fewer than min_count times.
      4. Sort vocabulary by frequency (descending) and assign integer ids.
      5. Convert the token stream to id integers (unknown words dropped).
      6. Save word_to_id mapping to vocab.json for later reuse.

    Why sort by frequency descending?
      The negative-sampling table is built from word_freqs indexed by word id.
      Having the most common words at the front is a convention that makes
      the CDF array slightly faster to search (not strictly required).

    Args:
        file_path:  path to the text8 file.
        min_count:  discard words with fewer than this many occurrences.

    Returns:
        token_ids  (np.ndarray, int32): the corpus as a sequence of word ids.
        word_to_id (dict):              word string -> integer id.
        id_to_word (dict):              integer id  -> word string.
        word_freqs (np.ndarray, float): raw count for each word id, shape (V,).
    """
    text = load_data(file_path)
    tokens = tokenise(text)
    word_count = get_word_count(tokens)

    # Filter rare words and sort by frequency descending
    vocab = [(w, c) for w, c in word_count.items() if c >= min_count]
    vocab.sort(key=lambda x: -x[1])

    word_to_id = {w: i for i, (w, _) in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    word_freqs = np.array([c for _, c in vocab], dtype=np.float64)

    token_ids = np.array(
        [word_to_id[w] for w in tokens if w in word_to_id],
        dtype=np.int32,
    )

    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(word_to_id, f)

    print(f"  Vocabulary size : {len(vocab):>8,}  (min_count={min_count})")
    print(f"  Token ids       : {len(token_ids):>8,}")

    return token_ids, word_to_id, id_to_word, word_freqs


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_tokens(token_ids, word_freqs, t=1e-5, rng=None):
    """Discard frequent words with a probability that grows with their frequency.

    Mikolov et al. (2013) proposed keeping word w with probability:

        P(keep | w) = min( sqrt( t / f(w) ), 1 )

    where f(w) = count(w) / total_tokens is the relative frequency of w
    and t is a small threshold (typically 1e-5).

    Intuition: very common words like "the" or "a" co-occur with almost every
    word in the corpus. Each occurrence carries little new information, so we
    can safely discard most of them. This speeds up training AND improves
    embedding quality for content words because the model is no longer
    overwhelmed by high-frequency noise.

    Example: a word with f=0.01 (1% of corpus) has P(keep) ≈ 3.2% when t=1e-5.

    Args:
        token_ids:  np.ndarray of integer word ids.
        word_freqs: np.ndarray of raw counts, indexed by word id, shape (V,).
        t:          threshold parameter (paper default 1e-5).
        rng:        numpy Generator, optional.

    Returns:
        Filtered list of integer token ids.
    """
    if rng is None:
        rng = np.random.default_rng()

    total     = word_freqs.sum()
    rel_freq  = word_freqs / total           # relative frequency per word id
    keep_prob = np.sqrt(t / rel_freq)
    keep_prob = np.minimum(keep_prob, 1.0)   # rare words always kept

    rand_vals = rng.random(len(token_ids))
    kept = [tok for tok, r in zip(token_ids, rand_vals) if r < keep_prob[tok]]
    print(f"  After subsampling : {len(kept):>8,}  (from {len(token_ids):,})")
    return np.array(kept, dtype=np.int32)


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def build_neg_sampler(word_freqs, rng=None):
    """Return a function that draws noise words from the unigram^0.75 distribution.

    The original word2vec raises each word's raw frequency to the 3/4 power
    before normalising:

        P_noise(w) ∝ freq(w)^0.75

    Why 0.75? Raising to a power < 1 smooths the distribution relative to
    plain unigram: rare words become more likely to appear as negatives than
    under unigram sampling, which gives the model better gradient signal for
    learning their representations. The value 0.75 is empirical — the authors
    found it worked best on downstream tasks.

    Implementation: instead of a 100-million-slot lookup table (original C
    code), we store the CDF and use np.searchsorted. This is O(log V) per
    sample and statistically identical.

    Args:
        word_freqs: np.ndarray of raw counts indexed by word id, shape (V,).
        rng:        numpy Generator, optional.

    Returns:
        sample(n) — function that returns an np.ndarray of n noise word ids.
    """
    if rng is None:
        rng = np.random.default_rng()

    adjusted = word_freqs ** 0.75
    adjusted /= adjusted.sum()
    cdf = np.cumsum(adjusted)

    def sample(n):
        r = rng.random(n)
        return np.searchsorted(cdf, r).astype(np.int32)

    return sample
