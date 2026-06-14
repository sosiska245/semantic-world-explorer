"""Benchmark domain mode vs semantic / multivec / hybrid on 10 target queries.

Run:
    .venv/bin/python scripts/run_domain_audit.py
"""

import os
import sys

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

from src.data_loader import EMBEDDINGS, ENTITIES_DF, PROFILE_LENS
from src.similarity import embed_query

PENALTY_TARGET = 7000.0
PENALTY_TOP_EXEMPT = 5


def apply_penalty(scores):
    pen = np.minimum(1.0, PROFILE_LENS / PENALTY_TARGET)
    exempt = np.argsort(scores)[::-1][:PENALTY_TOP_EXEMPT]
    pen[exempt] = 1.0
    return scores * pen


def semantic_scores(vec):
    return apply_penalty(EMBEDDINGS @ vec)


def multivec_scores(vec, text):
    from src.multivec import multi_sim, apply_length_penalty

    _ENERGY_TERMS = {
        "energy", "solar", "renewable", "electricity", "nuclear",
        "hydroelectric", "photovoltaic", "clean energy", "fossil fuel",
    }
    alpha = 0.10 if any(t in text.lower() for t in _ENERGY_TERMS) else 0.30
    s = multi_sim(vec, alpha=alpha)
    return apply_length_penalty(s)


def hybrid_scores(vec, text):
    from src.bm25 import blend as bm25_blend
    return bm25_blend(multivec_scores(vec, text), text, alpha=0.08)


def domain_scores(vec):
    from src.domain_sim import domain_sim
    from src.multivec import apply_length_penalty
    return apply_length_penalty(domain_sim(vec, alpha=0.4))


ISO3 = ENTITIES_DF["iso3"].tolist()
NAMES = ENTITIES_DF["name"].tolist()


def top_n(scores, n=20):
    idx = np.argsort(scores)[::-1][:n]
    return [(ISO3[i], NAMES[i], float(scores[i])) for i in idx]


def hits_at_k(scores, expected, k=20):
    top = {ISO3[i] for i in np.argsort(scores)[::-1][:k]}
    return sum(1 for iso3 in expected if iso3 in top)


QUERIES = [
    # (query_text, category, expected_iso3_list)
    ("arms weapons export manufacturer",
     "military",
     ["USA", "RUS", "FRA", "DEU", "GBR", "ISR", "CHN", "ITA", "ESP", "KOR"]),

    ("military spending armed forces power",
     "military",
     ["USA", "CHN", "RUS", "SAU", "IND", "GBR", "DEU", "FRA", "JPN", "KOR"]),

    ("NATO military alliance member",
     "military",
     ["USA", "GBR", "DEU", "FRA", "POL", "NOR", "TUR", "ITA", "NLD", "ESP"]),

    ("peacekeeping UN military troops",
     "military",
     ["ETH", "BGD", "PAK", "RWA", "GHA", "IND", "NGA", "FRA", "URY", "NLD"]),

    ("aviation airports air travel hub",
     "aviation",
     ["NLD", "GBR", "ARE", "SGP", "DEU", "USA", "JPN", "TUR", "FRA", "KOR"]),

    ("major airline international hub airport",
     "aviation",
     ["ARE", "SGP", "GBR", "NLD", "USA", "DEU", "TUR", "JPN", "FRA", "KOR"]),

    ("semiconductor chip manufacturing technology",
     "technology",
     ["TWN", "KOR", "JPN", "USA", "DEU", "NLD", "CHN", "MYS", "SGP", "IRL"]),

    ("artificial intelligence research technology",
     "technology",
     ["USA", "CHN", "GBR", "CAN", "DEU", "FRA", "ISR", "JPN", "KOR", "SGP"]),

    ("medical tourism destination healthcare",
     "tourism_health",
     ["THA", "IND", "MYS", "SGP", "ISR", "MEX", "TUR", "POL", "CZE", "HUN"]),

    ("UNESCO cultural heritage tourism",
     "tourism_health",
     ["ITA", "ESP", "FRA", "CHN", "IND", "MEX", "GRC", "EGY", "JPN", "PRT"]),
]


def run():
    print("=" * 70)
    print("DOMAIN MODE BENCHMARK — 10 target queries")
    print("Modes: Semantic | Multivec | Hybrid | Domain")
    print("=" * 70)

    totals = {"semantic": 0, "multivec": 0, "hybrid": 0, "domain": 0}
    category_results = {}

    for q_text, cat, expected in QUERIES:
        print(f"\n>>> [{cat}] \"{q_text}\"")
        print(f"    Expected (top-10 subset): {expected[:6]}...")

        vec = embed_query(q_text)

        s_scores = semantic_scores(vec)
        m_scores = multivec_scores(vec, q_text)
        h_scores = hybrid_scores(vec, q_text)
        d_scores = domain_scores(vec)

        s_hit = hits_at_k(s_scores, expected)
        m_hit = hits_at_k(m_scores, expected)
        h_hit = hits_at_k(h_scores, expected)
        d_hit = hits_at_k(d_scores, expected)

        n_exp = len(expected)
        print(f"    Hits@20/{n_exp}: Semantic={s_hit}  Multivec={m_hit}  Hybrid={h_hit}  Domain={d_hit}")

        best = max(s_hit, m_hit, h_hit, d_hit)
        winner = [k for k, v in [("S", s_hit), ("M", m_hit), ("H", h_hit), ("D", d_hit)] if v == best]
        print(f"    Best: {'/'.join(winner)} ({best}/{n_exp})")

        # Show domain top-5 for context
        d_top5 = top_n(d_scores, 5)
        s_top5 = top_n(s_scores, 5)
        print(f"    Domain top-5: {[f'{n}({i})' for i, n, _ in d_top5]}")
        print(f"    Semantic top-5: {[f'{n}({i})' for i, n, _ in s_top5]}")

        totals["semantic"] += s_hit
        totals["multivec"] += m_hit
        totals["hybrid"]   += h_hit
        totals["domain"]   += d_hit

        category_results.setdefault(cat, []).append((q_text, s_hit, m_hit, h_hit, d_hit, n_exp))

    total_expected = sum(len(exp) for _, _, exp in QUERIES)
    print("\n" + "=" * 70)
    print("OVERALL HITS@20")
    print("=" * 70)
    for mode, total in totals.items():
        pct = 100 * total / total_expected
        bar = "#" * int(pct / 2)
        print(f"  {mode:9s}: {total:3d}/{total_expected}  ({pct:.1f}%)  {bar}")

    print("\nBY CATEGORY:")
    for cat, rows in category_results.items():
        cat_exp = sum(r[5] for r in rows)
        cat_s = sum(r[1] for r in rows)
        cat_m = sum(r[2] for r in rows)
        cat_h = sum(r[3] for r in rows)
        cat_d = sum(r[4] for r in rows)
        print(f"  {cat:15s}: S={cat_s}/{cat_exp}  M={cat_m}/{cat_exp}  H={cat_h}/{cat_exp}  D={cat_d}/{cat_exp}")

    print("\nDOMAIN vs BEST-PRIOR (max of S/M/H) per query:")
    for q_text, cat, expected in QUERIES:
        vec = embed_query(q_text)
        s = hits_at_k(semantic_scores(vec), expected)
        m = hits_at_k(multivec_scores(vec, q_text), expected)
        h = hits_at_k(hybrid_scores(vec, q_text), expected)
        d = hits_at_k(domain_scores(vec), expected)
        best_prior = max(s, m, h)
        diff = d - best_prior
        sign = "+" if diff > 0 else ""
        print(f"  {sign}{diff:+2d}  [{cat}] {q_text[:50]}")


if __name__ == "__main__":
    run()
