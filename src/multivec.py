"""Multi-section similarity: for each country, blend main embedding score with
the best-matching section embedding score. Additive — main embedding unchanged."""

import numpy as np
from src.data_loader import EMBEDDINGS, SECTION_EMBEDDINGS, N_ENTITIES, PROFILE_LENS
from src.config import EMBEDDING_DIM

# Suppress short-profile "hub" countries (e.g. Saint Martin 665 chars, Åland 5197)
# that score moderately for unrelated queries due to low profile specificity.
# Countries in the raw top-5 are exempt so legitimate small islands still surface.
_PENALTY_TARGET = 7000.0  # chars at which penalty = 1.0
_PENALTY_TOP_EXEMPT = 5   # protect top-N by raw score from suppression


def apply_length_penalty(scores: np.ndarray) -> np.ndarray:
    pen = np.minimum(1.0, PROFILE_LENS / _PENALTY_TARGET)
    exempt = np.argsort(scores)[::-1][: _PENALTY_TOP_EXEMPT]
    pen[exempt] = 1.0
    return scores * pen


def _precompute_matrix(key):
    vecs = SECTION_EMBEDDINGS.get(key, [None] * N_ENTITIES)
    mat = np.zeros((N_ENTITIES, EMBEDDING_DIM), dtype=np.float32)
    has_data = np.zeros(N_ENTITIES, dtype=bool)
    for i, v in enumerate(vecs):
        if v is not None:
            mat[i] = v
            has_data[i] = True
    return mat, has_data


_MATS = {k: _precompute_matrix(k) for k in SECTION_EMBEDDINGS}


def multi_sim(query_vec: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """alpha controls section boost weight. Falls back to pure main if no sections."""
    main = EMBEDDINGS @ query_vec

    any_sections = any(has.any() for _, has in _MATS.values())
    if not any_sections:
        return main

    section_maxes = np.full(N_ENTITIES, -np.inf, dtype=np.float32)
    has_any = np.zeros(N_ENTITIES, dtype=bool)
    for key, (mat, has_data) in _MATS.items():
        if not has_data.any():
            continue
        s = mat @ query_vec
        improved = has_data & (s > section_maxes)
        section_maxes[improved] = s[improved]
        has_any[improved] = True

    result = main.copy()
    result[has_any] = (1 - alpha) * main[has_any] + alpha * section_maxes[has_any]
    return result
