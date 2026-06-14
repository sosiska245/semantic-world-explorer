"""Runtime similarity: embed short user queries (Voyage, query mode, cached)
and compare against precomputed document embeddings via dot product (Voyage
embeddings are pre-normalized, so dot product == cosine similarity).
"""

import os

import numpy as np
from dotenv import load_dotenv

from src.cache import query_embedding_cache
from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, SLOT_COLORS

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        import voyageai

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set. Copy .env.example to .env and fill it in.")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_query(text):
    """Return a normalized embedding vector for a short query string, or None
    if text is empty. Cached per normalized text."""
    text = (text or "").strip()
    if not text:
        return None

    key = text.lower()
    cached = query_embedding_cache.get(key)
    if cached is not None:
        return cached

    client = _get_client()
    result = client.embed(
        [text], model=EMBEDDING_MODEL, input_type="query", output_dimension=EMBEDDING_DIM
    )
    vec = np.array(result.embeddings[0], dtype=np.float32)
    query_embedding_cache.set(key, vec)
    return vec


def sims_from_store_data(sim_data, n_entities):
    """Convert a dcc.Store payload ({"R": [floats]|None, ...}) back into
    {"R": np.ndarray|None, ...}."""
    sims = {}
    for color in SLOT_COLORS:
        vals = (sim_data or {}).get(color)
        sims[color] = np.array(vals, dtype=np.float32) if vals is not None else None
    return sims
