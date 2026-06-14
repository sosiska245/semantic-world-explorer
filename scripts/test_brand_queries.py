"""
Targeted brand-query test against the current emb_brand embeddings in parquet.

Tests 15 culturally specific queries where brand profiles should excel and
the main embedding typically fails. For each query, shows:
  - top-10 countries (brand score for profiled, main score for others)
  - rank of the expected country
  - brand score vs main score for expected country
  - visual indicator of success/failure

Usage:
    .venv/bin/python scripts/test_brand_queries.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

import voyageai
client = voyageai.Client()

# ── Load data ─────────────────────────────────────────────────────────────────
PARQUET_PATH = os.path.join(ROOT_DIR, "data", "processed", "embeddings.parquet")
df = pd.read_parquet(PARQUET_PATH)

N = len(df)
ISO3  = df["iso3"].tolist()
NAMES = df["name"].tolist()

# Main embeddings (all rows, normalised)
MAIN_MAT = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
norms = np.linalg.norm(MAIN_MAT, axis=1, keepdims=True)
MAIN_MAT = MAIN_MAT / np.where(norms > 0, norms, 1.0)

# Brand embeddings (sparse — None where no profile)
BRAND_VECS: dict[int, np.ndarray] = {}
for i, v in enumerate(df["emb_brand"]):
    if v is not None:
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            BRAND_VECS[i] = v / norm

print(f"Loaded: {N} entities, {len(BRAND_VECS)} with brand profiles.")

# ── Score function ────────────────────────────────────────────────────────────
def brand_scores(qvec: np.ndarray) -> np.ndarray:
    """
    Returns an N-length score array.
    Countries with brand profiles: brand cosine.
    Countries without: main cosine (fallback).
    """
    main = MAIN_MAT @ qvec
    scores = main.copy()
    for i, bv in BRAND_VECS.items():
        scores[i] = float(np.dot(qvec, bv))
    return scores


def embed_query(text: str) -> np.ndarray:
    resp = client.embed([text], model="voyage-3", input_type="query")
    v = np.array(resp.embeddings[0], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ── Test queries ──────────────────────────────────────────────────────────────
TESTS = [
    ("anime",                     "JPN",  "entertainment"),
    ("manga",                     "JPN",  "entertainment"),
    ("K-pop Korean music",        "KOR",  "music"),
    ("reggae music",              "JAM",  "music"),
    ("tango dance",               "ARG",  "music"),
    ("flamenco",                  "ESP",  "music"),
    ("Formula One racing",        "GBR",  "sports"),
    ("Ferrari sports car",        "ITA",  "brand"),
    ("LEGO toy bricks",           "DNK",  "brand"),
    ("IKEA furniture",            "SWE",  "brand"),
    ("sushi Japanese food",       "JPN",  "food"),
    ("coffee origin Ethiopia",    "ETH",  "food"),
    ("champagne sparkling wine",  "FRA",  "food"),
    ("Bollywood film industry",   "IND",  "film"),
    ("Nollywood African cinema",  "NGA",  "film"),
]

# ── Run tests ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BRAND QUERY VALIDATION — 15 culturally specific queries")
print("Brand mode: brand cosine for profiled countries, main cosine for others")
print("=" * 80)

passes = 0
top3   = 0
top10  = 0

for query, expected_iso, category in TESTS:
    qvec = embed_query(query)
    scores = brand_scores(qvec)
    ranked = np.argsort(scores)[::-1]

    expected_idx = ISO3.index(expected_iso) if expected_iso in ISO3 else -1
    expected_rank = int(np.where(ranked == expected_idx)[0][0]) + 1 if expected_idx >= 0 else -1

    # Compare brand vs main for expected country
    main_sim  = float(MAIN_MAT[expected_idx] @ qvec) if expected_idx >= 0 else 0.0
    brand_sim = float(np.dot(BRAND_VECS[expected_idx], qvec)) if expected_idx in BRAND_VECS else None
    has_brand = expected_idx in BRAND_VECS

    result_icon = "✓✓" if expected_rank <= 3 else ("✓" if expected_rank <= 10 else "✗")
    if expected_rank <= 3:  top3 += 1
    if expected_rank <= 10: top10 += 1
    if expected_rank <= 20: passes += 1

    # Top-5 names
    top5 = [NAMES[ranked[i]] for i in range(5)]

    brand_str = f"{brand_sim:.3f}" if brand_sim is not None else "n/a (no profile)"
    print(f"\n[{result_icon}] [{category}] '{query}'")
    print(f"    Expected: {expected_iso} ({NAMES[ISO3.index(expected_iso)] if expected_iso in ISO3 else '?'}) → rank {expected_rank}")
    print(f"    Main sim: {main_sim:.3f}  |  Brand sim: {brand_str}  |  Has profile: {has_brand}")
    print(f"    Top-5: {', '.join(top5)}")

print("\n" + "=" * 80)
print(f"SUMMARY: {len(TESTS)} queries")
print(f"  top-3  hits: {top3}/{len(TESTS)}  ({100*top3/len(TESTS):.0f}%)")
print(f"  top-10 hits: {top10}/{len(TESTS)}  ({100*top10/len(TESTS):.0f}%)")
print(f"  top-20 hits: {passes}/{len(TESTS)}  ({100*passes/len(TESTS):.0f}%)")
print("=" * 80)
