"""Domain knowledge similarity: blends main embedding with best matching
domain-specific embedding (military / aviation / technology / tourism_health).

Domain embeddings are built from structured factual texts (World Bank data,
NATO/nuclear static lists, UNESCO counts) — separate from Wikipedia section
embeddings used by multivec.py.

Usage:
    from src.domain_sim import domain_sim
    scores = domain_sim(query_vec, alpha=0.4)
"""

import numpy as np
from src.data_loader import DOMAIN_EMBEDDINGS, EMBEDDINGS, N_ENTITIES

DOMAIN_KEYS = ["military", "aviation", "technology", "tourism_health"]

# Precompute (matrix, has_data_mask) per domain key at import time.
_DOMAIN_MATS = {}
for _key in DOMAIN_KEYS:
    _entry = DOMAIN_EMBEDDINGS.get(_key)
    if _entry is not None:
        _mat, _mask = _entry
        _DOMAIN_MATS[_key] = (_mat, _mask)


def domain_sim(query_vec: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend main embedding score with the best-matching domain score.

    alpha=0 → pure main embedding (identical to semantic mode).
    alpha=1 → pure best-domain score (ignores main embedding).
    Default alpha=0.4: 60% main + 40% best domain.

    Countries with no domain data (zero vectors) are unaffected — their
    domain score will be 0.0 and has_any stays False, so they fall back
    to the pure main embedding score.
    """
    main = EMBEDDINGS @ query_vec

    domain_maxes = np.full(N_ENTITIES, -np.inf, dtype=np.float32)
    has_any = np.zeros(N_ENTITIES, dtype=bool)

    for key, (mat, mask) in _DOMAIN_MATS.items():
        if not mask.any():
            continue
        scores = mat @ query_vec
        better = mask & (scores > domain_maxes)
        domain_maxes[better] = scores[better]
        has_any[better] = True

    result = main.copy()
    result[has_any] = (1.0 - alpha) * main[has_any] + alpha * domain_maxes[has_any]
    return result
