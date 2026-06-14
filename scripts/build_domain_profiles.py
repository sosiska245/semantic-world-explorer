"""Build domain-specific embedding columns and add them to embeddings.parquet.

For each country, builds 4 short structured domain texts and embeds them
via Voyage AI, adding columns emb_domain_{military,aviation,technology,
tourism_health} to data/processed/embeddings.parquet.

Prerequisites:
  - scripts/fetch_domain_data.py already run (data/interim/domain_facts.json)
  - data/processed/embeddings.parquet exists
  - data/processed/facts_by_country.csv exists

Run:
  .venv/bin/python scripts/build_domain_profiles.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import voyageai
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL  # noqa: E402

DOMAIN_FACTS_JSON  = os.path.join(ROOT_DIR, "data", "interim", "domain_facts.json")
OWID_FACTS_CSV     = os.path.join(ROOT_DIR, "data", "processed", "facts_by_country.csv")
EMBEDDINGS_PARQUET = os.path.join(ROOT_DIR, "data", "processed", "embeddings.parquet")

BATCH_SIZE = 10
SECONDS_BETWEEN_REQUESTS = 2
RATE_LIMIT_RETRY = 65

DOMAIN_COLS = [
    "emb_domain_military",
    "emb_domain_aviation",
    "emb_domain_technology",
    "emb_domain_tourism_health",
]

# Hardcoded supplements for countries absent from World Bank data (non-UN members etc.)
WB_SUPPLEMENTS = {
    # Taiwan — key for semiconductor/technology queries
    "TWN": {
        "military_personnel": (163000, 2023),
        "rd_pct_gdp":         (3.50,   2022),
        "hitech_exports_pct": (52.0,   2022),
    },
    # Kosovo
    "XKX": {
        "military_personnel": (5000, 2022),
    },
}

# Air passengers per 1000 people: approximate for key hub countries not covered
# by WB air_passengers / World Bank population ratio
PAX_PER_1000_OVERRIDES = {
    "SGP": 10400,  # ~58M pax, 5.5M people
    "ARE": 8700,   # ~86M pax (DXB+AUH+SHJ), 9.9M people
    "QAT": 5800,   # ~32M pax, 2.9M people (DOH)
    "ISL": 4200,   # ~10M pax, 0.37M people (KEF)
    "NLD": 3900,   # ~71M pax, 17.9M people (AMS)
    "HKG": 10000,  # ~71M pax (pre-COVID), 7.4M people — not in our 250 but just in case
    "LUX": 2000,   # ~4M pax, 0.66M people
    "AUT": 1600,   # ~14M pax, 9M people (VIE)
    "FIN": 1500,   # ~8M pax, 5.5M people (HEL)
    "NOR": 1400,   # ~8M pax, 5.4M people (OSL)
    "TUR": 1100,   # ~95M pax (pre-COVID), 84M people (IST/SAW)
}


def _fmt_int(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f} million"
    if v >= 1_000:
        return f"{v/1_000:.0f} thousand"
    return str(int(v))


def build_military_text(name, iso3, domain_facts, owid_by_iso3):
    nato    = domain_facts["nato"]
    nuclear = domain_facts["nuclear"]
    ns      = domain_facts["nato_nuclear_sharing"]
    wb      = domain_facts["wb"]

    parts = [f"{name} (military/defence)."]

    # NATO / alliance status
    if iso3 in nato:
        parts.append("NATO member.")
    # Nuclear status
    if iso3 in nuclear:
        parts.append(f"Nuclear weapons: {nuclear[iso3]}.")
    elif iso3 in ns:
        parts.append("Hosts US nuclear weapons under NATO nuclear sharing.")

    # Military expenditure from OWID facts
    owid = owid_by_iso3.get(iso3, {})
    if "military_expenditure_gdp" in owid:
        v, yr = owid["military_expenditure_gdp"]
        parts.append(f"Military expenditure: {v:.1f}% of GDP ({yr}).")

    # Active personnel from World Bank
    mil_data = wb.get("military_personnel", {}).get(iso3)
    if mil_data:
        v, yr = mil_data
        parts.append(f"Active military personnel: approximately {_fmt_int(v)} ({yr}).")

    if len(parts) == 1:
        return ""   # no data at all
    return " ".join(parts)


def build_aviation_text(name, iso3, domain_facts, owid_by_iso3, pop_by_iso3):
    wb = domain_facts["wb"]

    parts = [f"{name} (aviation/transport)."]

    owid = owid_by_iso3.get(iso3, {})
    pax_total = None
    pax_yr    = None
    if "air_passengers" in owid:
        pax_total, pax_yr = owid["air_passengers"]

    # Passengers per 1000 people
    pax_per_1000 = PAX_PER_1000_OVERRIDES.get(iso3)
    if pax_per_1000 is None and pax_total is not None:
        pop = pop_by_iso3.get(iso3)
        if pop and pop > 0:
            pax_per_1000 = int(round(pax_total / pop * 1000))

    if pax_total is not None:
        parts.append(
            f"Air passengers: approximately {pax_total/1e6:.1f}M total ({pax_yr})"
            + (f", approximately {pax_per_1000:,} per 1,000 people." if pax_per_1000 else ".")
        )
    elif pax_per_1000:
        parts.append(f"Air passengers: approximately {pax_per_1000:,} per 1,000 people.")

    # Logistics Performance Index
    lpi_data = wb.get("lpi", {}).get(iso3)
    if lpi_data:
        v, yr = lpi_data
        parts.append(f"Logistics Performance Index: {v:.1f}/5.0 ({yr}).")

    if len(parts) == 1:
        return ""
    return " ".join(parts)


def build_technology_text(name, iso3, domain_facts, owid_by_iso3):
    wb   = domain_facts["wb"]
    owid = owid_by_iso3.get(iso3, {})

    parts = [f"{name} (technology/innovation)."]

    rd_data = wb.get("rd_pct_gdp", {}).get(iso3) or WB_SUPPLEMENTS.get(iso3, {}).get("rd_pct_gdp")
    if rd_data:
        v, yr = rd_data
        parts.append(f"R&D expenditure: {v:.2f}% of GDP ({yr}).")

    if "internet_users_share" in owid:
        v, yr = owid["internet_users_share"]
        parts.append(f"Internet users: {v:.0f}% of population ({yr}).")

    ht_data = (
        wb.get("hitech_exports_pct", {}).get(iso3)
        or WB_SUPPLEMENTS.get(iso3, {}).get("hitech_exports_pct")
    )
    if ht_data:
        v, yr = ht_data
        parts.append(f"High-technology exports: {v:.0f}% of manufactured exports ({yr}).")

    if len(parts) == 1:
        return ""
    return " ".join(parts)


def build_tourism_health_text(name, iso3, domain_facts, owid_by_iso3):
    owid    = owid_by_iso3.get(iso3, {})
    unesco  = domain_facts["unesco"].get(iso3, 0)

    parts = [f"{name} (tourism/health)."]

    if "tourist_arrivals" in owid:
        v, yr = owid["tourist_arrivals"]
        parts.append(f"International tourist arrivals: approximately {v:,.0f} ({yr}).")

    if unesco > 0:
        parts.append(f"UNESCO World Heritage Sites: {unesco}.")

    if "life_expectancy" in owid:
        v, yr = owid["life_expectancy"]
        parts.append(f"Life expectancy: {v:.0f} years ({yr}).")

    if "health_expenditure_gdp" in owid:
        v, yr = owid["health_expenditure_gdp"]
        parts.append(f"Healthcare spending: {v:.1f}% of GDP ({yr}).")

    if len(parts) == 1:
        return ""
    return " ".join(parts)


def embed_batch(client, texts):
    while True:
        try:
            result = client.embed(
                texts, model=EMBEDDING_MODEL,
                input_type="document", output_dimension=EMBEDDING_DIM,
            )
            return result.embeddings
        except voyageai.error.RateLimitError:
            print(f"    rate limited, sleeping {RATE_LIMIT_RETRY}s")
            time.sleep(RATE_LIMIT_RETRY)
        except Exception as exc:
            print(f"    transient error ({exc}), retrying 30s")
            time.sleep(30)


def main():
    load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("VOYAGE_API_KEY not set")

    print("Loading data...")
    with open(DOMAIN_FACTS_JSON) as f:
        domain_facts = json.load(f)

    owid_df = pd.read_csv(OWID_FACTS_CSV)
    # Index: iso3 -> {indicator: (value, year)}
    owid_by_iso3 = {}
    for iso3, grp in owid_df.groupby("iso3"):
        owid_by_iso3[iso3] = {}
        for _, row in grp.iterrows():
            try:
                v = float(row["value"])
            except (ValueError, TypeError):
                v = row["value"]
            owid_by_iso3[iso3][row["indicator"]] = (v, int(row["year"]))

    # Apply WB supplements to domain_facts wb dict
    for iso3, sups in WB_SUPPLEMENTS.items():
        for key, val in sups.items():
            if iso3 not in domain_facts["wb"].get(key, {}):
                domain_facts["wb"].setdefault(key, {})[iso3] = val

    df = pd.read_parquet(EMBEDDINGS_PARQUET)

    # Population lookup from profile_text stats or from OWID population_density proxy
    # Use WB population from military_personnel denominator — but easier: use rough numbers
    # from existing stats. We have population in stats_by_country.csv but not in parquet.
    # Approximate: use land_area + pop_density from OWID if available.
    pop_by_iso3 = {}
    for iso3, facts in owid_by_iso3.items():
        if "land_area_km2" in facts and "population_density" in facts:
            area  = facts["land_area_km2"][0]
            dens  = facts["population_density"][0]
            pop_by_iso3[iso3] = area * dens

    print("Building domain texts...")
    domain_texts = {col: [] for col in DOMAIN_COLS}
    counts = {col: 0 for col in DOMAIN_COLS}

    for _, row in df.iterrows():
        iso3 = row["iso3"]
        name = row["name"]

        mil  = build_military_text(name, iso3, domain_facts, owid_by_iso3)
        avi  = build_aviation_text(name, iso3, domain_facts, owid_by_iso3, pop_by_iso3)
        tech = build_technology_text(name, iso3, domain_facts, owid_by_iso3)
        tou  = build_tourism_health_text(name, iso3, domain_facts, owid_by_iso3)

        domain_texts["emb_domain_military"].append(mil)
        domain_texts["emb_domain_aviation"].append(avi)
        domain_texts["emb_domain_technology"].append(tech)
        domain_texts["emb_domain_tourism_health"].append(tou)

        for col, t in [("emb_domain_military", mil), ("emb_domain_aviation", avi),
                       ("emb_domain_technology", tech), ("emb_domain_tourism_health", tou)]:
            if t:
                counts[col] += 1

    for col, n in counts.items():
        print(f"  {col}: {n}/250 non-empty")

    print(f"\nEmbedding via Voyage AI ({EMBEDDING_MODEL})...")
    client = voyageai.Client(api_key=api_key)

    for col in DOMAIN_COLS:
        texts = domain_texts[col]
        print(f"\n  {col}")

        # Build placeholder embeddings (zero vector for empty texts)
        all_embs = [None] * len(texts)
        todo_idx = [i for i, t in enumerate(texts) if t]
        todo_texts = [texts[i] for i in todo_idx]
        print(f"    {len(todo_texts)} non-empty texts to embed")

        for start in range(0, len(todo_texts), BATCH_SIZE):
            batch = todo_texts[start: start + BATCH_SIZE]
            batch_idx = todo_idx[start: start + BATCH_SIZE]
            embs = embed_batch(client, batch)
            for idx, emb in zip(batch_idx, embs):
                all_embs[idx] = list(emb)
            print(f"    {min(start + BATCH_SIZE, len(todo_texts))}/{len(todo_texts)}")
            if start + BATCH_SIZE < len(todo_texts):
                time.sleep(SECONDS_BETWEEN_REQUESTS)

        # Fill empty texts with zero vectors
        zero = [0.0] * EMBEDDING_DIM
        for i in range(len(all_embs)):
            if all_embs[i] is None:
                all_embs[i] = zero

        df[col] = all_embs
        df.to_parquet(EMBEDDINGS_PARQUET, index=False)
        print(f"    saved {col} to parquet")

    print(f"\nDone. {EMBEDDINGS_PARQUET} now has columns:")
    final_df = pd.read_parquet(EMBEDDINGS_PARQUET)
    dom_cols = [c for c in final_df.columns if c.startswith("emb_domain_")]
    for c in dom_cols:
        non_zero = sum(1 for v in final_df[c] if v is not None and any(x != 0 for x in v))
        print(f"  {c}: {non_zero}/250 non-zero")


if __name__ == "__main__":
    main()
