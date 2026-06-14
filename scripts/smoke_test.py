"""Automated smoke tests. Exit 0 = all pass. Run after each rebuild."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
errors = []

def check(condition, msg):
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")

print("=== smoke_test.py ===")

# 1. Parquet shape
try:
    import pandas as pd, numpy as np
    df = pd.read_parquet("data/processed/embeddings.parquet")
    check(len(df) == 250, f"embeddings 250 rows (got {len(df)})")
    check("embedding" in df.columns, "embedding column present")
    check("iso3" in df.columns, "iso3 column present")
    check(len(df["embedding"].iloc[0]) == 1024, "embedding dim=1024")
    print(f"  info: columns: {list(df.columns)}")
except Exception as e:
    errors.append(f"embeddings.parquet load failed: {e}")

# 2. Facts CSV categories
try:
    facts = pd.read_csv("data/processed/facts_by_country.csv")
    for cat in ["energy", "agriculture", "politics", "geography"]:
        check(cat in facts["category"].values, f"facts category={cat}")
    print(f"  info: facts categories: {sorted(facts['category'].unique())}")
    print(f"  info: facts rows: {len(facts)}, countries: {facts['iso3'].nunique()}")
except Exception as e:
    errors.append(f"facts CSV load failed: {e}")

# 3. Module imports
try:
    from src.data_loader import ENTITIES_DF, EMBEDDINGS, N_ENTITIES
    check(N_ENTITIES == 250, f"N_ENTITIES=250 (got {N_ENTITIES})")
    check(EMBEDDINGS.shape == (250, 1024), f"EMBEDDINGS (250,1024) got {EMBEDDINGS.shape}")
    from src.facts_loader import FACTS_BY_COUNTRY, match_category
    check(len(FACTS_BY_COUNTRY) > 0, f"FACTS_BY_COUNTRY non-empty ({len(FACTS_BY_COUNTRY)})")
    check(match_category("renewable energy") == "energy", "match_category energy")
    check(match_category("democracy") == "politics", "match_category politics")
except Exception as e:
    errors.append(f"module import failed: {e}")

# 4. BM25 (only if module exists)
if os.path.exists("src/bm25.py"):
    try:
        from src.bm25 import bm25_scores, blend as bm25_blend
        from src.data_loader import N_ENTITIES, ENTITIES_DF
        scores = bm25_scores("renewable energy")
        check(len(scores) == N_ENTITIES, "bm25_scores correct length")
        check(scores.max() > 0, "bm25_scores non-zero")
        fjord = bm25_scores("fjords")
        nor_idx = ENTITIES_DF[ENTITIES_DF["iso3"] == "NOR"].index
        if len(nor_idx):
            rank = int((fjord > fjord[nor_idx[0]]).sum()) + 1
            check(rank <= 10, f"Norway 'fjords' BM25 rank={rank} (want <=10)")
    except Exception as e:
        errors.append(f"bm25 module: {e}")

# 5. Section embeddings (only if columns present)
try:
    df = pd.read_parquet("data/processed/embeddings.parquet")
    sec_cols = [c for c in df.columns if c.startswith("emb_")]
    if sec_cols:
        print(f"  info: section columns: {sec_cols}")
        for col in sec_cols:
            nn = df[col].notna().sum()
            check(nn > 100, f"{col} >100 non-null (got {nn})")
    else:
        print("  info: no section embedding columns yet")
except Exception as e:
    errors.append(f"section emb check: {e}")

# 6. Flagged query ranks (semantic, uses Voyage API — skipped if no key)
try:
    from src.data_loader import ENTITIES_DF, EMBEDDINGS
    from src.similarity import embed_query
    FLAGGED = [
        ("aviation and airports", "NLD", 50),
        ("aviation and airports", "ISL", 50),
        ("agriculture economy",   "BRA", 60),
        ("music festivals culture","GBR", 60),
    ]
    print("  Flagged query ranks (semantic):")
    for query, iso3, threshold in FLAGGED:
        try:
            vec = embed_query(query)
            sims = EMBEDDINGS @ vec
            rows = ENTITIES_DF[ENTITIES_DF["iso3"] == iso3]
            if rows.empty: continue
            idx = rows.index[0]
            rank = int((sims > sims[idx]).sum()) + 1
            tag = "ok" if rank <= threshold else "WARN"
            print(f"    [{tag}] {query!r} -> {iso3} rank={rank} (want <={threshold})")
        except Exception as e:
            print(f"    [skip] {query!r}: {e}")
except Exception as e:
    print(f"  [skip] flagged query test: {e}")

print("")
if errors:
    print(f"SMOKE TEST FAILED — {len(errors)} error(s):")
    for e in errors: print(f"  - {e}")
    sys.exit(1)
else:
    print("SMOKE TEST PASSED")
    sys.exit(0)
