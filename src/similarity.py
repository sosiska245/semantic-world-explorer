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


def normalize_channel(sim):
    """Min-max normalize a similarity array to [0, 1] for color contrast.
    Raw cosine similarities between a short query and long profiles cluster
    in a narrow band (~0.2-0.6), so normalization is needed for visible
    color contrast on the map."""
    if sim is None:
        return None
    lo, hi = float(np.min(sim)), float(np.max(sim))
    if hi - lo < 1e-9:
        return np.zeros_like(sim)
    return (sim - lo) / (hi - lo)


def normalize_all(sims, n_entities):
    """sims: dict color -> raw similarity array or None.
    Returns dict color -> np.ndarray[n_entities] normalized to [0, 1]
    (zeros for inactive slots)."""
    out = {}
    for color in SLOT_COLORS:
        normalized = normalize_channel(sims.get(color))
        out[color] = normalized if normalized is not None else np.zeros(n_entities)
    return out


NEUTRAL_GRAY = "rgb(120,120,140)"


def colors_from_normalized(normalized, mode, n_entities):
    """normalized: dict color -> np.ndarray[n] in [0, 1] (e.g. from
    normalize_all, possibly subset to fewer rows via fancy indexing).
    mode: "blend" for an RGB blend of all three channels, or "R"/"G"/"B"
    for a single-channel view (used by the city drill-down map).
    Returns list[str] of 'rgb(r,g,b)' values, one per entity."""
    zeros = np.zeros(n_entities, dtype=int)
    if mode == "blend":
        r = (normalized["R"] * 255).astype(int)
        g = (normalized["G"] * 255).astype(int)
        b = (normalized["B"] * 255).astype(int)
    else:
        intensity = (normalized[mode] * 255).astype(int)
        if mode == "R":
            r, g, b = intensity, zeros, zeros
        elif mode == "G":
            r, g, b = zeros, intensity, zeros
        else:
            r, g, b = zeros, zeros, intensity
    return [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]
