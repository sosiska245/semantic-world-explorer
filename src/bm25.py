"""BM25 index over country profile texts.
Built at startup from ENTITIES_DF. Used when ranking mode = 'hybrid'."""

import re
import numpy as np
from rank_bm25 import BM25Okapi
from src.data_loader import ENTITIES_DF

_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","has","have","had","it","its","this","that",
    "by","from","as","be","been","which","who","their","also","than","more",
    "into","about","between","after","before","during","under","over",
}

def _tokenize(text):
    tokens = re.findall(r"\b[a-z]+\b", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

_corpus = [_tokenize(t) for t in ENTITIES_DF["profile_text"]]
_BM25 = BM25Okapi(_corpus)

def bm25_scores(query: str) -> np.ndarray:
    tokens = _tokenize(query)
    if not tokens:
        return np.zeros(len(ENTITIES_DF), dtype=np.float32)
    return np.array(_BM25.get_scores(tokens), dtype=np.float32)

def blend(cosine_sim: np.ndarray, query: str, alpha: float = 0.08) -> np.ndarray:
    """alpha=0 pure cosine, alpha=1 pure BM25. BM25 is min-max normalized first."""
    bm25 = bm25_scores(query)
    b_min, b_max = bm25.min(), bm25.max()
    if b_max - b_min < 1e-9:
        return cosine_sim
    b_norm = (bm25 - b_min) / (b_max - b_min)
    return (1.0 - alpha) * cosine_sim + alpha * b_norm
