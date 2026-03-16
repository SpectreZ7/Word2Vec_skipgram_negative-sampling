

import numpy as np

# ---------------------------------------------------------------------------
# Device-agnostic sigmoid
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid that works with both NumPy and CuPy arrays.

    Uses cupy.get_array_module(x) to select the right library at runtime,
    so the same function works whether x lives on CPU or GPU.

    For x >= 0: 1 / (1 + exp(-x))    — avoids overflow in exp(-x)
    For x <  0: exp(x) / (1 + exp(x)) — avoids overflow in exp(-x) for large |x|
    """
    try:
        import cupy
        xp = cupy.get_array_module(x)
    except ImportError:
        xp = np

    return xp.where(
        x >= 0,
        1.0 / (1.0 + xp.exp(-x)),
        xp.exp(x) / (1.0 + xp.exp(x)),
    )


# ---------------------------------------------------------------------------
# Skip-gram pair generation (illustrative — not used in the main training loop)
# ---------------------------------------------------------------------------
# The training loop in train.py generates pairs inline to avoid materialising
# ~170M pairs in memory. This function shows the concept clearly.

def generate_skipgram_pairs(token_ids, max_window_size):
    """Generate (center_id, context_id) pairs from a token sequence.

    The window size is re-sampled uniformly from [1, max_window_size] for
    each center word. This is the original word2vec trick: it gives words
    closer to the center a higher effective weight because they appear in
    more window configurations (closer words fall inside more sampled windows).

    Args:
        token_ids:       list of integer token ids.
        max_window_size: maximum context window radius (e.g. 5).

    Returns:
        List of (center_id, context_id) integer tuples.
    """
    rng   = np.random.default_rng()
    pairs = []
    n     = len(token_ids)

    for center_pos in range(n):
        center_id   = token_ids[center_pos]
        window_size = rng.integers(1, max_window_size + 1)

        left  = max(0, center_pos - window_size)
        right = min(n, center_pos + window_size + 1)

        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue
            context_id = token_ids[context_pos]
            pairs.append((center_id, context_id))

    return pairs


# ---------------------------------------------------------------------------
# Word2Vec model
# ---------------------------------------------------------------------------

class Word2Vec:
    """Skip-gram Word2Vec with negative sampling.

    Supports both CPU (NumPy) and GPU (CuPy) via the `xp` parameter.

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary (V).
    embed_dim : int
        Dimensionality of the embedding vectors (d).
    xp : module, optional
        Array module to use — either `numpy` or `cupy`.
        Defaults to cupy if available, otherwise numpy.

    Attributes
    ----------
    W_in  : ndarray, shape (V, d)
        Input (center-word) embedding matrix.
        Initialised uniform in [−0.5/d, 0.5/d] following the original paper.
    W_out : ndarray, shape (V, d)
        Output (context-word) embedding matrix.
        Initialised to zeros so all scores start neutral.
    """

    def __init__(self, vocab_size, embed_dim=100, xp=None):
        if xp is None:
            try:
                import cupy
                xp = cupy
            except ImportError:
                xp = np

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.xp         = xp

        self.W_in  = (xp.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
        self.W_out = xp.zeros((vocab_size, embed_dim), dtype=xp.float64)

    # -----------------------------------------------------------------------
    # Single-pair forward pass (CPU-friendly reference implementation)
    # -----------------------------------------------------------------------

    def forward(self, center_id, context_id, neg_ids):
        """Compute the SGNS loss for one (center, context) pair.

        Args:
            center_id:  int
            context_id: int
            neg_ids:    array of ints, shape (K,)

        Returns:
            loss: float. The value −L; lower is better.
        """
        u_c   = self.W_in[center_id]
        v_o   = self.W_out[context_id]
        v_neg = self.W_out[neg_ids]

        s_pos = self.xp.dot(v_o, u_c)
        s_neg = v_neg @ u_c

        loss = -self.xp.log(sigmoid(s_pos) + 1e-10) \
               - self.xp.sum(self.xp.log(sigmoid(-s_neg) + 1e-10))

        return float(loss)

    # -----------------------------------------------------------------------
    # Single-pair backward + SGD (keep for reference and CPU smoke-testing)
    # -----------------------------------------------------------------------

    def backward_and_update(self, center_id, context_id, neg_ids, lr):
        """Compute gradients and apply an SGD update for one pair.

        See module docstring for full gradient derivation.

        Args:
            center_id:  int
            context_id: int
            neg_ids:    array of ints, shape (K,)
            lr:         float, learning rate.

        Returns:
            loss: float, −L before the update (for logging).
        """
        xp    = self.xp
        u_c   = self.W_in[center_id]
        v_o   = self.W_out[context_id]
        v_neg = self.W_out[neg_ids]

        s_pos = xp.dot(v_o, u_c)
        s_neg = v_neg @ u_c

        sig_pos = sigmoid(s_pos)
        sig_neg = sigmoid(s_neg)

        loss = -xp.log(sig_pos + 1e-10) - xp.sum(xp.log(1.0 - sig_neg + 1e-10))

        e_pos = 1.0 - sig_pos
        e_neg = -sig_neg

        grad_u_c   = e_pos * v_o + e_neg @ v_neg
        grad_v_o   = e_pos * u_c
        grad_v_neg = xp.outer(e_neg, u_c)

        self.W_in[center_id]   += lr * grad_u_c
        self.W_out[context_id] += lr * grad_v_o
        self.W_out[neg_ids]    += lr * grad_v_neg

        return float(loss)

    # -----------------------------------------------------------------------
    # Batched backward + SGD (GPU-efficient)
    # -----------------------------------------------------------------------

    def backward_and_update_batch(self, center_ids, context_ids, neg_ids, lr):
        """Compute gradients and apply SGD for a batch of B pairs at once.

        This is the same math as backward_and_update, broadcast over B pairs.
        Processing a batch in one call amortises GPU kernel-launch overhead,
        giving orders-of-magnitude speedup over one-pair-at-a-time on GPU.

        Args:
            center_ids:  array of ints, shape (B,)  — center word indices
            context_ids: array of ints, shape (B,)  — true context indices
            neg_ids:     array of ints, shape (B, K) — noise word indices
            lr:          float, learning rate

        Returns:
            loss: float, mean −L over the batch (for logging).

        Gradient derivation (batched)
        -----------------------------
        For each pair i in [0, B):
            s_pos[i]   = V_pos[i] · U[i]
            s_neg[i,k] = V_neg[i,k] · U[i]

        Error signals (same as single-pair):
            e_pos[i]   = 1 − σ(s_pos[i])   ← how wrong is the positive score?
            e_neg[i,k] = −σ(s_neg[i,k])    ← how wrong is each negative score?

        Gradients:
            grad_U[i]       = e_pos[i] * V_pos[i]  +  Σ_k e_neg[i,k] * V_neg[i,k]
            grad_V_pos[i]   = e_pos[i] * U[i]
            grad_V_neg[i,k] = e_neg[i,k] * U[i]

        xp.add.at for scatter-add
        -------------------------
        Standard `W_in[center_ids] += ...` is buffered: if center_ids has
        duplicates, only the last update is applied (numpy silently drops others).
        xp.add.at is unbuffered: every gradient contribution is applied, even for
        repeated indices. This is the correct behaviour for mini-batch SGD.
        """
        xp = self.xp
        d  = self.embed_dim

        U     = self.W_in[center_ids]              # (B, d)
        V_pos = self.W_out[context_ids]             # (B, d)
        V_neg = self.W_out[neg_ids]                 # (B, K, d)

        # Scores: element-wise dot products over the d dimension
        s_pos = (U * V_pos).sum(axis=1)             # (B,)
        s_neg = (V_neg * U[:, None, :]).sum(axis=2) # (B, K)

        sig_pos = sigmoid(s_pos)                    # (B,)
        sig_neg = sigmoid(s_neg)                    # (B, K)

        # Loss (mean over batch, for logging)
        loss = (
            -xp.log(sig_pos + 1e-10).mean()
            - xp.log(1.0 - sig_neg + 1e-10).mean()
        )

        # Error signals
        e_pos = 1.0 - sig_pos                       # (B,)
        e_neg = -sig_neg                             # (B, K)

        # Gradients
        grad_U     = (e_pos[:, None] * V_pos
                      + (e_neg[:, :, None] * V_neg).sum(axis=1))  # (B, d)
        grad_V_pos = e_pos[:, None] * U                            # (B, d)
        grad_V_neg = e_neg[:, :, None] * U[:, None, :]             # (B, K, d)

        # Scatter-add updates (handles duplicate indices correctly)
        xp.add.at(self.W_in,  center_ids,        lr * grad_U)
        xp.add.at(self.W_out, context_ids,        lr * grad_V_pos)
        xp.add.at(self.W_out, neg_ids.ravel(),    lr * grad_V_neg.reshape(-1, d))

        return float(loss)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_embedding(self, word_id):
        """Return the W_in embedding for a word id as a CPU numpy array."""
        v = self.W_in[word_id]
        return np.array(v) if not isinstance(v, np.ndarray) else v.copy()

    def save(self, path):
        """Save W_in to a .npy file. Transfers from GPU to CPU if needed."""
        W = self.W_in
        if hasattr(W, 'get'):      # CuPy arrays have a .get() method
            W = W.get()
        np.save(path, W)
        print(f"Embeddings saved to {path}  shape={W.shape}")

    def load(self, path):
        """Load W_in from a .npy file (loads to CPU; move to GPU manually if needed)."""
        self.W_in = np.load(path)
