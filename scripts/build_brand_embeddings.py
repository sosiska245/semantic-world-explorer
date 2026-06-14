"""
Build brand profile embeddings and add emb_brand column to parquet.

Reads brand_profiles.json (or a batch file), embeds each profile in document
mode via Voyage, then writes/updates the emb_brand column in embeddings.parquet.

Safe to re-run: skips countries that already have emb_brand set in parquet.
Pass --batch PATH to use a separate batch JSON file (merged into main after).

Usage:
    .venv/bin/python scripts/build_brand_embeddings.py
    .venv/bin/python scripts/build_brand_embeddings.py --batch data/processed/brand_profiles_b2.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

import voyageai

MAIN_PROFILES = os.path.join(ROOT_DIR, "data", "processed", "brand_profiles.json")
PARQUET_PATH  = os.path.join(ROOT_DIR, "data", "processed", "embeddings.parquet")
BATCH_SIZE    = 25
MODEL         = "voyage-4"
OUTPUT_DIM    = 1024

client = voyageai.Client()


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    resp = client.embed(texts, model=MODEL, input_type="document", output_dimension=OUTPUT_DIM)
    return [np.array(v, dtype=np.float32) for v in resp.embeddings]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=None, help="Path to batch JSON file to merge and embed")
    args = parser.parse_args()

    # ── Load main profiles ────────────────────────────────────────────────────
    with open(MAIN_PROFILES) as f:
        main_profiles: dict[str, str] = json.load(f)

    # ── Merge batch file if provided ──────────────────────────────────────────
    if args.batch:
        with open(args.batch) as f:
            batch: dict[str, str] = json.load(f)
        new_keys = [k for k in batch if k not in main_profiles]
        main_profiles.update(batch)
        with open(MAIN_PROFILES, "w") as f:
            json.dump(main_profiles, f, indent=2, ensure_ascii=False)
        print(f"Merged {len(batch)} profiles from {args.batch} ({len(new_keys)} new) → brand_profiles.json")
        profiles = batch   # only embed the batch (new ones)
    else:
        profiles = main_profiles

    print(f"Loaded {len(profiles)} profiles from brand_profiles.json")

    # ── Load parquet ──────────────────────────────────────────────────────────
    df = pd.read_parquet(PARQUET_PATH)
    iso3_to_idx = {row["iso3"]: i for i, row in df.iterrows()}
    print(f"Parquet: {len(df)} rows. Known ISOs in dataset: {len(iso3_to_idx)}")

    # Initialise column to None if not present
    if "emb_brand" not in df.columns:
        df["emb_brand"] = None
        print("  Created new emb_brand column (all None).")
    else:
        existing = df["emb_brand"].notna().sum()
        print(f"  emb_brand column exists — {existing} rows already have embeddings.")

    # ── Filter: must be in dataset AND not already embedded ───────────────────
    already = set()
    if "emb_brand" in df.columns:
        for _, row in df.iterrows():
            if row["emb_brand"] is not None:
                already.add(row["iso3"])

    valid = {
        iso: txt for iso, txt in profiles.items()
        if iso in iso3_to_idx and iso not in already
    }
    skipped_missing = [iso for iso in profiles if iso not in iso3_to_idx]
    skipped_existing = [iso for iso in profiles if iso in already]
    if skipped_missing:
        print(f"  Skipping {len(skipped_missing)} ISOs not in dataset: {skipped_missing}")
    if skipped_existing:
        print(f"  Skipping {len(skipped_existing)} already embedded: {skipped_existing[:5]}{'…' if len(skipped_existing)>5 else ''}")
    if not valid:
        print("  Nothing new to embed — all profiles already in parquet.")
        return
    print(f"  Embedding {len(valid)} new profiles …")

    # ── Embed in batches ──────────────────────────────────────────────────────
    iso3s = list(valid.keys())
    texts = list(valid.values())
    all_vecs: dict[str, np.ndarray] = {}

    for batch_start in range(0, len(iso3s), BATCH_SIZE):
        batch_iso  = iso3s[batch_start : batch_start + BATCH_SIZE]
        batch_text = texts[batch_start : batch_start + BATCH_SIZE]
        print(f"    Batch {batch_start // BATCH_SIZE + 1}: {batch_iso[0]} … {batch_iso[-1]}", end=" ", flush=True)
        t0 = time.time()
        vecs = embed_batch(batch_text)
        elapsed = time.time() - t0
        for iso, vec in zip(batch_iso, vecs):
            all_vecs[iso] = vec
        print(f"({elapsed:.1f}s)")
        if batch_start + BATCH_SIZE < len(iso3s):
            time.sleep(0.5)   # gentle rate limit

    print(f"  Embedded {len(all_vecs)} profiles successfully.")

    # ── Write to dataframe ────────────────────────────────────────────────────
    for iso, vec in all_vecs.items():
        idx = iso3_to_idx[iso]
        df.at[idx, "emb_brand"] = vec

    filled = df["emb_brand"].notna().sum()
    print(f"\n  emb_brand filled: {filled}/{len(df)} rows")

    # ── Save parquet ──────────────────────────────────────────────────────────
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"  Saved → {PARQUET_PATH}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    df2 = pd.read_parquet(PARQUET_PATH)
    v = df2[df2["iso3"] == "JPN"]["emb_brand"].iloc[0]
    print(f"\n  Sanity check — JPN emb_brand: type={type(v).__name__}, shape={v.shape}, norm={np.linalg.norm(v):.4f}")


if __name__ == "__main__":
    main()
