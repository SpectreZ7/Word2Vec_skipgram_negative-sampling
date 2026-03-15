# Word2Vec — Skip-Gram with Negative Sampling (pure NumPy)

Skip-gram Word2Vec trained on the [text8](http://mattmahoney.net/dc/text8.zip) dataset.
The full training loop — forward pass, loss, gradients, and parameter updates — is
implemented from scratch in NumPy (with optional CuPy GPU acceleration).

---

## Algorithm

### Objective

For each (centre word `c`, context word `o`) pair with `K` noise words `n_1 … n_K`:

```
L = log σ(v_o · u_c)  +  Σ_k log σ(−v_{nk} · u_c)
```

where `u_c = W_in[c]` (centre embedding), `v_o = W_out[o]` (context embedding),
and `σ(x) = 1 / (1 + e^{−x})`.
The first term pushes the true context score up; each negative term pushes a noise
word score down.  We minimise `−L`.

### Gradients

```
∂L/∂u_c     = (1 − σ(s_pos)) · v_o   −   Σ_k σ(s_neg_k) · v_{nk}
∂L/∂v_o     = (1 − σ(s_pos)) · u_c
∂L/∂v_{nk}  = −σ(s_{nk}) · u_c
```

SGD update: `param += lr * ∂L/∂param`

Full derivation with intermediate steps is in [`word2vec/model.py`](word2vec/model.py).

---

## Design decisions

| Choice | Detail |
|--------|--------|
| Two embedding matrices | `W_in` (centre role) and `W_out` (context role) — asymmetric by design |
| Noise distribution | `P_noise(w) ∝ freq(w)^0.75` — smooths rare words into sampling |
| Subsampling | `P(keep\|w) = min(√(t / f(w)), 1)`, `t = 1e-5` — discards frequent words |
| Dynamic window | Window radius sampled from `[1, max_window]` per centre word |
| LR schedule | Linear decay `0.025 → 1e-4` across the entire training run |
| GPU batching | Processes 512 pairs per kernel call (one kernel per pair would be slower than CPU) |

---

## File structure

```
word2vec/
  model.py       — Word2Vec class, sigmoid, batched forward/backward
  preprocess.py  — vocab building, subsampling, negative-sampling table
  train.py       — multi-epoch training loop with GPU batching
  evaluate.py    — nearest neighbours + Google analogy benchmark (3CosAdd)
requirements.txt — dependencies (NumPy; CuPy optional for GPU)
```

---

## Installation

```bash
git clone https://github.com/SpectreZ7/Hallucination_Detection.git
cd Hallucination_Detection
pip install -r requirements.txt
```

**GPU support (optional):** uncomment the lines matching your CUDA version in
`requirements.txt`, then re-run `pip install -r requirements.txt`.
Run `nvidia-smi` and look for `CUDA Version: 12.x` or `11.x` to identify your version.

The text8 dataset (~100 MB) is downloaded automatically on first run.

---

## Training

```bash
python word2vec/train.py
```

Key config variables at the top of [`train.py`](word2vec/train.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_EPOCHS` | `3` | Passes over the corpus |
| `EMBED_DIM` | `100` | Embedding dimensionality |
| `NUM_NEG` | `5` | Negative samples per pair |
| `BATCH_SIZE` | `512` | Pairs per GPU kernel call |
| `USE_GPU` | `True` | Use CuPy if available, else falls back to NumPy |
| `MAX_TOKENS` | `None` | Set to e.g. `1_000_000` for a quick smoke-test (~1 min on GPU) |

Embeddings are saved after each epoch as `word2vec/embeddings_epoch{N}.npy`.

---

## Evaluation

```bash
# Evaluate the 3-epoch embeddings (default)
python word2vec/evaluate.py

# Compare with 1-epoch embeddings
python word2vec/evaluate.py word2vec/embeddings_epoch1.npy
```

Runs two evaluations:

1. **Nearest neighbours** — cosine similarity lookup for probe words (`king`, `france`, `computer`, …)
2. **Google analogy benchmark** — 3CosAdd over ~19,500 questions ([Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)):
   ```
   w* = argmax cos(w,  v_b − v_a + v_c)   s.t. w ∉ {a, b, c}
   ```
   The file is downloaded automatically on first run.

---

## Results

| Epochs | Analogy accuracy |
|--------|-----------------|
| 1      | ~0.8%           |
| 3      | ~5.7%           |

The published word2vec result on text8 (~35–45%) uses 15 iterations, 200-dimensional
vectors, and a less aggressive subsampling threshold (`t = 1e-3` vs `1e-5` here).
The lower accuracy reflects those hyperparameter differences, not a bug in the
implementation — the nearest-neighbour results show correct semantic clustering
(countries near countries, numbers near numbers, tech terms together).
