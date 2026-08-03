# Semantic Search

A small semantic search engine built for Lab 3: it turns a corpus of short text
documents into embeddings, lets you search the corpus by meaning (not just
keyword matching) using cosine similarity, and visualizes the embedding space
with PCA.

## What it does

1. **Embeds every document** in `documents.json` into a vector (`embeddings.py`).
2. **Stacks them** into one NumPy matrix, one row per document (`search.py`).
3. **Searches** a query by embedding it, computing cosine similarity against
   every document, and returning the `top_k` highest-scoring matches (`search()`).
4. **Visualizes** the embedding space in 2D with PCA, colored by topic
   (`semantic_search_starter.ipynb`).

## Files

- `embeddings.py` — embedding logic. Two modes, selected by the
  `LAB3_EMBEDDING_MODE` environment variable:
  - `offline` (default) — a hashed bag-of-words vector. No API key, no network.
  - `api` — real embeddings from NVIDIA NIM (`nvidia/nv-embedqa-e5-v5`).
- `search.py` — document loading, embedding-matrix construction with a persisted
  cache, cosine similarity, and top-k search.
- `documents.json` — the corpus (20+ documents across 5 distinct topics).
- `embeddings_cache.json` — generated on first run; embeddings are loaded from
  here on later runs instead of being re-fetched.
- `semantic_search_starter.ipynb` — notebook that runs the pipeline, shows
  example queries, and produces the PCA scatter plot.
- `requirements.txt` — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

The API key comes from your Lab 1 NVIDIA NIM key. Copy `.env.example` to `.env`
and fill it in:

```bash
cp .env.example .env
```

## Running

### Offline mode (default, no key needed)

```bash
LAB3_EMBEDDING_MODE=offline python search.py
```

This runs the whole pipeline with no API key and no network, using the hashed
bag-of-words fallback. Good for testing.

### API mode (real semantic embeddings)

Set the mode in your `.env` file (or as an environment variable):

```bash
LAB3_EMBEDDING_MODE=api python search.py
```

Make sure `NVIDIA_API_KEY` is set in `.env`.

The full workflow — example queries and the PCA plot — lives in
`semantic_search_starter.ipynb`; run its cells top to bottom.

## Notes

- Embeddings are cached in `embeddings_cache.json`, keyed by embedding mode, so
  switching between `offline` and `api` doesn't reuse the wrong kind of vector.
- The offline fallback matches on shared vocabulary, not meaning, so its search
  results and clusters are rougher than API mode. That's expected — switch to
  API mode for real semantic results.
