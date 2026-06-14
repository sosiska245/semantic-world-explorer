"""Validate factual ranking vs embedding modes on numeric/factual queries.

Tests 8 queries where a direct indicator sort should theoretically outperform
embedding similarity. Compares: Semantic | Multivec | Hybrid | Domain | Factual.

Run:
    .venv/bin/python scripts/run_factual_rank_audit.py
"""

import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

from src.data_loader import EMBEDDINGS, ENTITIES_DF, PROFILE_LENS
from src.similarity import embed_query

ISO3  = ENTITIES_DF["iso3"].tolist()
NAMES = ENTITIES_DF["name"].tolist()
N     = len(ISO3)

PENALTY_TARGET   = 7000.0
PENALTY_EXEMPT   = 5
ENERGY_TERMS     = {"energy","solar","renewable","electricity","nuclear","hydroelectric"}


def _penalty(scores):
    pen = np.minimum(1.0, PROFILE_LENS / PENALTY_TARGET)
    pen[np.argsort(scores)[::-1][:PENALTY_EXEMPT]] = 1.0
    return scores * pen


def semantic(vec):
    return _penalty(EMBEDDINGS @ vec)


def multivec(vec, text):
    from src.multivec import multi_sim, apply_length_penalty
    alpha = 0.10 if any(t in text.lower() for t in ENERGY_TERMS) else 0.30
    return apply_length_penalty(multi_sim(vec, alpha=alpha))


def hybrid(vec, text):
    from src.bm25 import blend
    return blend(multivec(vec, text), text, alpha=0.08)


def domain(vec):
    from src.domain_sim import domain_sim
    from src.multivec import apply_length_penalty
    return apply_length_penalty(domain_sim(vec, alpha=0.4))


# ── Factual ranking ────────────────────────────────────────────────────────────

_facts_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "facts_by_country.csv"))

# Index: (iso3, indicator) → float value
_FACT_INDEX: dict = {}
for _, row in _facts_df.iterrows():
    key = (row["iso3"], row["indicator"])
    try:
        _FACT_INDEX[key] = float(row["value"])
    except (ValueError, TypeError):
        pass


def factual_rank(indicator: str, ascending: bool = False) -> np.ndarray:
    """Score array derived from indicator value. Countries with no data score 0."""
    scores = np.zeros(N, dtype=np.float32)
    for i, iso3 in enumerate(ISO3):
        val = _FACT_INDEX.get((iso3, indicator))
        if val is not None:
            scores[i] = val
    # Normalize to [0,1] for comparability; direction controlled by ascending
    vmin, vmax = scores.min(), scores.max()
    if vmax > vmin:
        norm = (scores - vmin) / (vmax - vmin)
        scores = (1.0 - norm) if ascending else norm
    return scores


def hits_at_k(scores, expected_iso3s, k=20):
    top_set = {ISO3[i] for i in np.argsort(scores)[::-1][:k]}
    return sum(1 for iso3 in expected_iso3s if iso3 in top_set)


def top5_names(scores):
    return [f"{NAMES[i]}({ISO3[i]})" for i in np.argsort(scores)[::-1][:5]]


# ── Query definitions ─────────────────────────────────────────────────────────
# Each entry: (query_text, indicator, ascending, expected_iso3s, note)

QUERIES = [

    # ── Group 1: Factual ranking expected to WIN ──────────────────────────────

    ("life expectancy longevity",
     "life_expectancy", False,
     ["JPN","CHE","ESP","ITA","AUS","SGP","ISL","LUX","NOR","SWE"],
     "Direct indicator: highest value = longest-lived countries"),

    ("internet users digital society",
     "internet_users_share", False,
     ["NOR","ISL","DNK","FIN","SWE","NLD","CHE","GBR","KOR","LUX"],
     "Direct indicator: highest % = most connected"),

    ("international tourist arrivals destination",
     "tourist_arrivals", False,
     ["FRA","ESP","USA","CHN","ITA","TUR","MEX","GBR","DEU","THA"],
     "Direct indicator: absolute arrivals ranks real tourism giants"),

    ("electoral democracy and political freedom",
     "democracy_index", False,
     ["NOR","FIN","SWE","DNK","NLD","NZL","CAN","AUS","CHE","DEU"],
     "Direct indicator: V-Dem index 0→1"),

    ("renewable energy leaders solar wind",
     "renewable_share", False,
     ["ISL","NOR","SWE","DNK","FIN","NZL","AUT","CHE","DEU","ESP"],
     "Direct indicator: share of renewables in energy mix (79 countries only)"),

    ("agriculture dominates economy farming",
     "agriculture_share_gdp", False,
     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"],
     "Direct indicator: agriculture as % of GDP"),

    # ── Group 2: Factual ranking expected to FAIL (wrong/proxy metric) ────────

    ("military spending armed forces power",
     "military_expenditure_gdp", False,
     ["USA","CHN","RUS","SAU","IND","GBR","DEU","FRA","JPN","KOR"],
     "PROXY MISMATCH: % of GDP ≠ absolute spending; USA(3.1%) loses to RUS(7.5%), ARM(6.1%)"),

    ("healthcare system universal coverage quality",
     "health_expenditure_gdp", False,
     ["NOR","SWE","FIN","AUS","CAN","DEU","FRA","JPN","NZL","CHE"],
     "PROXY MISMATCH: USA leads at 27.1% GDP but lacks universal coverage"),

]


def run():
    print("=" * 72)
    print("FACTUAL RANKING VALIDATION — 8 queries")
    print("Modes: Semantic | Multivec | Hybrid | Domain | Factual")
    print("=" * 72)

    totals = {m: 0 for m in ("semantic","multivec","hybrid","domain","factual")}
    group1_totals = {m: 0 for m in totals}
    group2_totals = {m: 0 for m in totals}

    for idx, (q_text, indicator, asc, expected, note) in enumerate(QUERIES):
        group = 1 if idx < 6 else 2
        print(f"\n[{idx+1}] \"{q_text}\"")
        print(f"    Indicator : {indicator} ({'ascending' if asc else 'descending'})")
        print(f"    Note      : {note}")
        n_exp = len(expected)

        vec = embed_query(q_text)

        s_sc = semantic(vec)
        m_sc = multivec(vec, q_text)
        h_sc = hybrid(vec, q_text)
        d_sc = domain(vec)
        f_sc = factual_rank(indicator, ascending=asc)

        s_h = hits_at_k(s_sc, expected)
        m_h = hits_at_k(m_sc, expected)
        h_h = hits_at_k(h_sc, expected)
        d_h = hits_at_k(d_sc, expected)
        f_h = hits_at_k(f_sc, expected)

        modes = [("Semantic", s_h), ("Multivec", m_h), ("Hybrid", h_h),
                 ("Domain", d_h), ("Factual", f_h)]
        best = max(v for _, v in modes)

        print(f"    Hits@20/{n_exp}: ", end="")
        for name, v in modes:
            marker = " ★" if v == best and v > 0 else ""
            print(f"{name}={v}{marker}  ", end="")
        print()

        # Top-5 for Factual and best embedding mode
        best_emb = max(("Semantic",s_h),("Multivec",m_h),("Hybrid",h_h),("Domain",d_h),
                       key=lambda x: x[1])
        best_emb_scores = {"Semantic":s_sc,"Multivec":m_sc,"Hybrid":h_sc,"Domain":d_sc}[best_emb[0]]
        print(f"    Factual top-5 : {top5_names(f_sc)}")
        print(f"    {best_emb[0]:8s} top-5 : {top5_names(best_emb_scores)}")

        for m, v in [("semantic",s_h),("multivec",m_h),("hybrid",h_h),
                     ("domain",d_h),("factual",f_h)]:
            totals[m] += v
            if group == 1:
                group1_totals[m] += v
            else:
                group2_totals[m] += v

    total_exp = sum(len(q[3]) for q in QUERIES)
    g1_exp    = sum(len(q[3]) for q in QUERIES[:6])
    g2_exp    = sum(len(q[3]) for q in QUERIES[6:])

    print("\n" + "=" * 72)
    print("OVERALL HITS@20")
    print("=" * 72)
    for m, v in totals.items():
        bar = "#" * int(100 * v / total_exp / 2)
        print(f"  {m:9s}: {v:3d}/{total_exp}  ({100*v/total_exp:.1f}%)  {bar}")

    print(f"\nGROUP 1 — Direct indicators (6 queries, {g1_exp} expected):")
    for m, v in group1_totals.items():
        print(f"  {m:9s}: {v}/{g1_exp}  ({100*v/g1_exp:.1f}%)")

    print(f"\nGROUP 2 — Proxy mismatch (2 queries, {g2_exp} expected):")
    for m, v in group2_totals.items():
        print(f"  {m:9s}: {v}/{g2_exp}  ({100*v/g2_exp:.1f}%)")

    print("\nFACTUAL vs BEST-EMBEDDING delta per query:")
    for q_text, indicator, asc, expected, _ in QUERIES:
        vec = embed_query(q_text)
        s_h = hits_at_k(semantic(vec), expected)
        m_h = hits_at_k(multivec(vec, q_text), expected)
        h_h = hits_at_k(hybrid(vec, q_text), expected)
        d_h = hits_at_k(domain(vec), expected)
        f_h = hits_at_k(factual_rank(indicator, asc), expected)
        best_emb = max(s_h, m_h, h_h, d_h)
        diff = f_h - best_emb
        sign = "+" if diff > 0 else ""
        print(f"  {sign}{diff:+2d}  {q_text[:55]}")

    print()


if __name__ == "__main__":
    run()
