"""Load the precomputed entity profiles + embeddings once at app startup.

data/processed/embeddings.parquet is the ONLY data file the running app
reads. It contains one row per country/city with a precomputed 1024-dim
document embedding (see scripts/build_embeddings.py). The app never embeds
profile text at runtime - only short user queries (see src/similarity.py).
"""

import os

import numpy as np
import pandas as pd

from src.config import EMBEDDINGS_PARQUET

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT_DIR, EMBEDDINGS_PARQUET)


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    embeddings = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    meta = df.drop(columns=["embedding"]).reset_index(drop=True)
    return meta, embeddings


ENTITIES_DF, EMBEDDINGS = load_data()
N_ENTITIES = len(ENTITIES_DF)

_ID_TO_INDEX = {entity_id: idx for idx, entity_id in enumerate(ENTITIES_DF["id"])}


def get_entity(entity_id):
    idx = _ID_TO_INDEX.get(entity_id)
    if idx is None:
        return None
    return ENTITIES_DF.iloc[idx].to_dict()


def get_entity_index(entity_id):
    return _ID_TO_INDEX.get(entity_id)


def entities_for_country(iso3, include_country=False):
    mask = ENTITIES_DF["iso3"] == iso3
    if not include_country:
        mask &= ENTITIES_DF["type"] == "city"
    return ENTITIES_DF[mask]
