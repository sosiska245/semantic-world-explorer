"""Brand / Association similarity.

Pure brand-profile cosine for countries with emb_brand set;
main embedding cosine fallback for countries without a profile.

Usage:
    from src.brand_sim import brand_sim
    scores = brand_sim(query_vec)
"""

import json
import os
import numpy as np

from src.data_loader import EMBEDDINGS, ENTITIES_DF, N_ENTITIES
from src.config import EMBEDDING_DIM

# Load profile char lengths for the length penalty.
_PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "brand_profiles.json")
with open(_PROFILES_PATH) as _f:
    _profile_chars: dict[str, int] = {iso: len(text) for iso, text in json.load(_f).items()}

# Precompute at import: row → unit-normed brand vector + light length penalty.
# Penalty: min(1.0, profile_chars / 1500)
# Suppresses short profiles (< 1500 chars) that dominate broad queries.
# Validated: zero regressions on 20 named-association wins; +3 H@5 on travel.
_BRAND_VECS: dict[int, np.ndarray] = {}
_BRAND_PENALTY: dict[int, float] = {}

_iso3_col = ENTITIES_DF["iso3"].tolist()

for _i, _v in enumerate(ENTITIES_DF["emb_brand"] if "emb_brand" in ENTITIES_DF.columns else []):
    if _v is not None:
        _arr = np.array(_v, dtype=np.float32)
        _n = np.linalg.norm(_arr)
        if _n > 1e-9:
            _BRAND_VECS[_i] = _arr / _n
            _chars = _profile_chars.get(_iso3_col[_i], 1500)
            _BRAND_PENALTY[_i] = min(1.0, _chars / 1500)

# Precomputed main-embedding matrix (already present in data_loader, reuse).
_MAIN_MAT = EMBEDDINGS.copy()
_norms = np.linalg.norm(_MAIN_MAT, axis=1, keepdims=True)
_MAIN_MAT = _MAIN_MAT / np.where(_norms > 0, _norms, 1.0)

BRAND_PROFILE_COUNT = len(_BRAND_VECS)


def brand_sim(query_vec: np.ndarray) -> np.ndarray:
    """Return N-length similarity array.

    Countries with brand profiles: brand cosine * light length penalty.
    Countries without: main embedding cosine fallback.
    """
    scores = _MAIN_MAT @ query_vec
    for i, bv in _BRAND_VECS.items():
        scores[i] = float(np.dot(bv, query_vec)) * _BRAND_PENALTY[i]
    return scores
