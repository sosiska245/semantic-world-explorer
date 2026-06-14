"""Embed every entity's profile_text via Voyage AI (document mode), batched
and resumable. This is the ONLY place document embeddings are computed - the
running app (src/similarity.py) only ever embeds short user queries at
runtime and compares them against these precomputed vectors.

Output: data/processed/embeddings.parquet
"""

import os
import shutil
import sys
import time

import pandas as pd
import voyageai
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL  # noqa: E402

INTERIM_PARQUET = os.path.join(ROOT_DIR, "data", "interim", "profiles.parquet")
OUT_PARQUET = os.path.join(ROOT_DIR, "data", "processed", "embeddings.parquet")

# data/interim/ is gitignored (intermediate pipeline output); copy the facts
# table to data/processed/ so it ships with the deployed app alongside
# embeddings.parquet. See src/facts_loader.py.
FACTS_INTERIM_CSV = os.path.join(ROOT_DIR, "data", "interim", "facts_by_country.csv")
FACTS_PROCESSED_CSV = os.path.join(ROOT_DIR, "data", "processed", "facts_by_country.csv")

# Payment method on file -> standard (tier 1) rate limits, far above the
# 3 RPM / 10K TPM free-tier cap. Batch of 10 max-length (~6000 char / ~2K
# token) profile texts is ~20K tokens per request, comfortably under tier-1
# TPM; short spacing keeps well under tier-1 RPM. RateLimitError retry stays
# as a safety net in case limits haven't kicked in yet.
BATCH_SIZE = 10
SECONDS_BETWEEN_REQUESTS = 2
RATE_LIMIT_RETRY_SECONDS = 65


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embed entity profiles via Voyage AI")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed all profiles even if already embedded (use after profile_text changes)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("VOYAGE_API_KEY not set. Copy .env.example to .env and fill it in.")

    df = pd.read_parquet(INTERIM_PARQUET)

    if os.path.exists(OUT_PARQUET) and not args.force:
        existing = pd.read_parquet(OUT_PARQUET)
        existing_ids = set(existing["id"])
    else:
        if args.force and os.path.exists(OUT_PARQUET):
            print("--force: clearing existing embeddings for full re-embed")
        existing = None
        existing_ids = set()

    todo = df[~df["id"].isin(existing_ids)].reset_index(drop=True)
    print(f"{len(existing_ids)} already embedded, {len(todo)} to embed")

    client = voyageai.Client(api_key=api_key)
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[start : start + BATCH_SIZE]
        texts = batch["profile_text"].tolist()

        while True:
            try:
                result = client.embed(
                    texts, model=EMBEDDING_MODEL, input_type="document", output_dimension=EMBEDDING_DIM
                )
                break
            except voyageai.error.RateLimitError as exc:
                print(f"  rate limited, sleeping {RATE_LIMIT_RETRY_SECONDS}s: {exc}")
                time.sleep(RATE_LIMIT_RETRY_SECONDS)

        new_rows = []
        for row, emb in zip(batch.to_dict("records"), result.embeddings):
            row["embedding"] = list(emb)
            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        if existing is not None:
            existing = pd.concat([existing, new_df], ignore_index=True)
        else:
            existing = new_df
        existing.to_parquet(OUT_PARQUET, index=False)

        print(f"  embedded {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")
        if start + BATCH_SIZE < len(todo):
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"Wrote {len(existing)} embedded profiles to {OUT_PARQUET}")

    # Merge text_* columns from profiles into existing (may be absent if main loop was skipped)
    text_cols = [c for c in df.columns if c.startswith("text_")]
    for col in text_cols:
        if col not in existing.columns:
            existing = existing.merge(df[["id", col]], on="id", how="left")
            print(f"  merged {col} from profiles into existing")

    SECTION_COLS = {
        "emb_economy":   "text_economy",
        "emb_culture":   "text_culture",
        "emb_geography": "text_geography",
        "emb_politics":  "text_politics",
        "emb_history":   "text_history",
    }

    for emb_col, text_col in SECTION_COLS.items():
        if text_col not in existing.columns:
            print(f"  {text_col} column not found in profiles, skipping {emb_col}")
            continue

        if emb_col not in existing.columns:
            existing[emb_col] = None

        if existing[emb_col].notna().all():
            print(f"  {emb_col} already complete, skipping")
            continue

        todo_mask = (
            existing[text_col].fillna("").str.len() > 50
        ) & existing[emb_col].isna()
        todo_idx = existing[todo_mask].index.tolist()
        print(f"  embedding {emb_col}: {len(todo_idx)} rows")

        for start in range(0, len(todo_idx), BATCH_SIZE):
            batch_idx = todo_idx[start : start + BATCH_SIZE]
            texts = existing.loc[batch_idx, text_col].tolist()

            while True:
                try:
                    result = client.embed(
                        texts, model=EMBEDDING_MODEL,
                        input_type="document", output_dimension=EMBEDDING_DIM,
                    )
                    break
                except voyageai.error.RateLimitError:
                    print(f"    rate limited, sleeping {RATE_LIMIT_RETRY_SECONDS}s")
                    time.sleep(RATE_LIMIT_RETRY_SECONDS)
                except Exception as exc:
                    print(f"    transient error ({exc}), retrying 30s")
                    time.sleep(30)

            for i, emb in zip(batch_idx, result.embeddings):
                existing.at[i, emb_col] = list(emb)

            existing.to_parquet(OUT_PARQUET, index=False)
            print(f"    {emb_col}: {min(start+BATCH_SIZE, len(todo_idx))}/{len(todo_idx)}")
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    if os.path.exists(FACTS_INTERIM_CSV):
        shutil.copy(FACTS_INTERIM_CSV, FACTS_PROCESSED_CSV)
        print(f"Copied facts table to {FACTS_PROCESSED_CSV}")


if __name__ == "__main__":
    main()
