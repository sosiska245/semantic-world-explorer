"""Shared constants used by both the offline embedding pipeline and the
runtime app. The embedding model/dimension MUST match between
scripts/build_embeddings.py (document embeddings) and src/similarity.py
(query embeddings) for dot-product similarity to be meaningful.
"""

EMBEDDING_MODEL = "voyage-4"
EMBEDDING_DIM = 1024

EMBEDDINGS_PARQUET = "data/processed/embeddings.parquet"

SLOT_COLORS = ["R", "G", "B"]
MAX_SLOTS = 3

SLOT_DISPLAY = {"R": "Slot 1 (Red)", "G": "Slot 2 (Green)", "B": "Slot 3 (Blue)"}
SLOT_COLOR_HEX = {"R": "#c2452d", "G": "#3f8f5c", "B": "#3b6ea5"}

SLOT_OPTIONS = [{"label": SLOT_DISPLAY[c], "value": c} for c in SLOT_COLORS]
