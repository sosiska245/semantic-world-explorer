"""Load the precomputed entity profiles + embeddings once at app startup.

data/processed/embeddings.parquet is the ONLY data file the running app
reads. It contains one row per country with a precomputed 1024-dim
document embedding (see scripts/build_embeddings.py). The app never embeds
profile text at runtime - only short user queries (see src/similarity.py).
"""

import os

import numpy as np
import pandas as pd

from src.config import EMBEDDING_DIM, EMBEDDINGS_PARQUET

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT_DIR, EMBEDDINGS_PARQUET)


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    embeddings = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    meta = df.drop(columns=["embedding"]).reset_index(drop=True)
    return meta, embeddings


ENTITIES_DF, EMBEDDINGS = load_data()
N_ENTITIES = len(ENTITIES_DF)
PROFILE_LENS = ENTITIES_DF["profile_text"].str.len().values.astype("float32")

_ID_TO_INDEX = {entity_id: idx for idx, entity_id in enumerate(ENTITIES_DF["id"])}

SECTION_KEYS = ["economy", "culture", "geography", "politics", "history"]

SECTION_EMBEDDINGS = {}
for _key in SECTION_KEYS:
    _col = f"emb_{_key}"
    if _col in ENTITIES_DF.columns:
        _vecs = []
        for _v in ENTITIES_DF[_col]:
            if _v is not None:
                _vecs.append(np.array(_v, dtype=np.float32))
            else:
                _vecs.append(None)
        SECTION_EMBEDDINGS[_key] = _vecs
    else:
        SECTION_EMBEDDINGS[_key] = [None] * N_ENTITIES


DOMAIN_KEYS = ["military", "aviation", "technology", "tourism_health"]

DOMAIN_EMBEDDINGS = {}
for _key in DOMAIN_KEYS:
    _col = f"emb_domain_{_key}"
    if _col in ENTITIES_DF.columns:
        _rows = []
        _mask = []
        for _v in ENTITIES_DF[_col]:
            if _v is not None and len(_v) > 0 and any(_x != 0 for _x in _v[:4]):
                _rows.append(np.array(_v, dtype=np.float32))
                _mask.append(True)
            else:
                _rows.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
                _mask.append(False)
        DOMAIN_EMBEDDINGS[_key] = (
            np.vstack(_rows).astype(np.float32),
            np.array(_mask, dtype=bool),
        )


def get_entity(entity_id):
    idx = _ID_TO_INDEX.get(entity_id)
    if idx is None:
        return None
    return ENTITIES_DF.iloc[idx].to_dict()


def get_entity_index(entity_id):
    return _ID_TO_INDEX.get(entity_id)


def similar_entities(entity_id, top_k=5):
    idx = _ID_TO_INDEX.get(entity_id)
    if idx is None:
        return []
    vec = EMBEDDINGS[idx]
    scores = EMBEDDINGS @ vec
    scores_copy = scores.copy()
    scores_copy[idx] = -1.0
    top_indices = np.argsort(scores_copy)[::-1][:top_k]
    return [(ENTITIES_DF.iloc[i]["name"], float(scores_copy[i])) for i in top_indices]
