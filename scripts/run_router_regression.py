"""Regression check for the Smart Query Router.

Verifies:
  - 20 broad semantic queries do NOT trigger factual routing
  - 10 factual queries DO trigger factual routing (correct indicator)
  - 5 edge cases do NOT trigger (military %, healthcare, borderline phrasing)

Also checks semantic fallback still produces reasonable results.

Run:
    .venv/bin/python scripts/run_router_regression.py
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
from src.factual_router import detect_route, factual_coverage, factual_scores
from src.similarity import embed_query

ISO3  = ENTITIES_DF["iso3"].tolist()
NAMES = ENTITIES_DF["name"].tolist()
N     = len(ISO3)

PENALTY_TARGET = 7000.0
PENALTY_EXEMPT = 5
ENERGY_TERMS   = {"energy","solar","renewable","electricity","nuclear","hydroelectric"}


def _penalty(scores):
    pen = np.minimum(1.0, PROFILE_LENS / PENALTY_TARGET)
    pen[np.argsort(scores)[::-1][:PENALTY_EXEMPT]] = 1.0
    return scores * pen


def semantic(vec, text):
    from src.multivec import apply_length_penalty, multi_sim
    alpha = 0.10 if any(t in text.lower() for t in ENERGY_TERMS) else 0.30
    return apply_length_penalty(multi_sim(vec, alpha=alpha))


def hits(scores, expected, k=20):
    top = {ISO3[i] for i in np.argsort(scores)[::-1][:k]}
    return sum(1 for iso3 in expected if iso3 in top)


def top5(scores):
    return [f"{NAMES[i]}({ISO3[i]})" for i in np.argsort(scores)[::-1][:5]]


# ── Query sets ────────────────────────────────────────────────────────────────

SHOULD_NOT_ROUTE = [
    # Broad cultural / geographic
    ("Buddhist culture and meditation temples",         ["THA","MMR","LKA","BTN","KHM","JPN","MNG"]),
    ("tropical rainforest biodiversity",                ["BRA","COD","IDN","PER","COL","ECU","CMR"]),
    ("alpine skiing winter sports mountains",           ["AUT","CHE","FRA","ITA","NOR","SWE","AND"]),
    ("oil and gas exporters petroleum",                 ["SAU","RUS","IRQ","IRN","ARE","KWT","NGA"]),
    ("football soccer culture passion",                 ["BRA","ARG","ESP","DEU","FRA","ITA","NLD"]),
    ("Viking Norse history heritage",                   ["NOR","SWE","DNK","ISL","FIN","GBR","IRL"]),
    ("ancient civilization archaeology ruins",          ["GRC","EGY","IRQ","IND","CHN","ITA","MEX"]),
    ("coffee production and culture",                   ["ETH","BRA","VNM","COL","IDN","UGA","MEX"]),
    ("fashion luxury brands design",                    ["FRA","ITA","USA","GBR","ESP","DEU","JPN"]),
    ("island nation ocean sea",                         ["MDV","FJI","PLW","FSM","WSM","TON","VUT"]),
    ("communist revolutionary socialist state",         ["CHN","CUB","PRK","VNM","LAO","BLR","VEN"]),
    ("desert nomad Bedouin camel arid",                 ["SAU","OMN","MAR","DZA","EGY","JOR","MRT"]),
    ("anime manga gaming esports",                      ["JPN","KOR","CHN","USA","FRA","DEU","SWE"]),
    ("wine viticulture vineyards grapes",               ["FRA","ITA","ESP","PRT","ARG","AUS","CHL"]),
    ("peacekeeping military UN troops",                 ["ETH","BGD","PAK","RWA","GHA","IND","NGA"]),
    ("semiconductor chip manufacturing",                ["TWN","KOR","JPN","USA","DEU","NLD","CHN"]),
    ("space exploration satellite technology",          ["USA","RUS","CHN","IND","FRA","JPN","ISR"]),
    ("coral reef marine diving scuba",                  ["AUS","PHL","IDN","MDV","FJI","PLW","MHL"]),
    ("authoritarian regime one-party state",            ["CHN","PRK","CUB","VEN","BLR","RUS","TKM"]),
    ("spicy street food cuisine flavours",              ["THA","IND","MEX","VNM","IDN","ETH","KOR"]),
]

SHOULD_ROUTE = [
    # (query, expected_indicator, expected_iso3s)
    ("life expectancy longevity",
     "life_expectancy",
     ["JPN","CHE","ESP","ITA","AUS","SGP","ISL","LUX","NOR","SWE"]),

    ("countries with longest life expectancy",
     "life_expectancy",
     ["JPN","CHE","ESP","ITA","AUS","SGP","ISL","LUX","NOR","SWE"]),

    ("international tourist arrivals destination",
     "tourist_arrivals",
     ["FRA","ESP","USA","CHN","ITA","TUR","MEX","GBR","DEU","THA"]),

    ("most visited countries most tourists",
     "tourist_arrivals",
     ["FRA","ESP","USA","CHN","ITA","TUR","MEX","GBR","DEU","THA"]),

    ("electoral democracy and political freedom",
     "democracy_index",
     ["NOR","FIN","SWE","DNK","NLD","NZL","CAN","AUS","CHE","DEU"]),

    ("liberal democracy rule of law",
     "democracy_index",
     ["NOR","FIN","SWE","DNK","NLD","NZL","CAN","AUS","CHE","DEU"]),

    ("internet users digital society",
     "internet_users_share",
     ["NOR","ISL","DNK","FIN","SWE","NLD","CHE","GBR","KOR","LUX"]),

    ("renewable energy leaders solar wind",
     "renewable_share",
     ["ISL","NOR","SWE","DNK","FIN","NZL","AUT","CHE","DEU","ESP"]),

    ("agriculture dominates economy farming",
     "agriculture_share_gdp",
     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"]),

    ("most agricultural economies farming gdp share",
     "agriculture_share_gdp",
     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"]),

    ("military spending defence budget",
     "military_spending_usd",
     ["USA","CHN","RUS","SAU","IND","GBR","DEU","FRA","JPN","KOR"]),

    ("biggest military spender country",
     "military_spending_usd",
     ["USA","CHN","RUS","SAU","IND","GBR","DEU","FRA","JPN","KOR"]),
]

EDGE_CASES_NO_ROUTE = [
    # Healthcare should NOT route (health_expenditure_gdp has small-island outliers)
    ("healthcare system universal coverage quality",  "NOT_ROUTED"),
    # Borderline phrasing — sound factual but shouldn't trigger
    ("solar power plant photovoltaic",                "NOT_ROUTED"),  # no "renewable energy leaders"
    ("how does democracy work political system",      "NOT_ROUTED"),  # query about democracy concept, not ranking
    ("agricultural country rice paddy field",         "NOT_ROUTED"),  # specific crop, not economy share
    ("peacekeeping military UN troops",               "NOT_ROUTED"),  # no trigger phrase
]


def run():
    print("=" * 70)
    print("ROUTER REGRESSION CHECK")
    print("=" * 70)

    # ── Group 1: Should NOT route ─────────────────────────────────────────
    print(f"\nGroup 1: Broad semantic queries (should NOT route) — {len(SHOULD_NOT_ROUTE)} queries")
    wrong_routes = []
    sem_hits_total = 0
    sem_exp_total  = 0
    for q, expected in SHOULD_NOT_ROUTE:
        route = detect_route(q)
        if route is not None:
            wrong_routes.append((q, route))
            marker = "  WRONG ROUTE →"
        else:
            marker = "  ok  "
        vec = embed_query(q)
        s_h = hits(semantic(vec, q), expected)
        sem_hits_total += s_h
        sem_exp_total  += len(expected)
        print(f"{marker} [{s_h}/{len(expected)} sem] {q[:55]}")

    if wrong_routes:
        print(f"\n  !! {len(wrong_routes)} UNEXPECTED ROUTES:")
        for q, r in wrong_routes:
            print(f"     '{q}' → {r[0]}")
    else:
        print(f"\n  All {len(SHOULD_NOT_ROUTE)} queries correctly fell through to semantic.")
    print(f"  Semantic fallback quality: {sem_hits_total}/{sem_exp_total} hits@20 ({100*sem_hits_total/sem_exp_total:.0f}%)")

    # ── Group 2: Should route ─────────────────────────────────────────────
    print(f"\nGroup 2: Factual queries (SHOULD route) — {len(SHOULD_ROUTE)} queries")
    missed_routes   = []
    wrong_indicator = []
    factual_hits_total = 0
    factual_exp_total  = 0
    sem_hits2_total    = 0
    for q, exp_ind, expected in SHOULD_ROUTE:
        route = detect_route(q)
        if route is None:
            missed_routes.append(q)
            marker = "  MISSED →"
            ind_ok = False
        else:
            ind_ok = route[0] == exp_ind
            if not ind_ok:
                wrong_indicator.append((q, exp_ind, route[0]))
            marker = f"  ok  [{route[0]}]" if ind_ok else f"  WRONG IND [{route[0]} ≠ {exp_ind}]"

        f_scores = factual_scores(route[0], ISO3, route[1]) if route else np.zeros(N, np.float32)
        vec = embed_query(q)
        f_h = hits(f_scores, expected)
        s_h = hits(semantic(vec, q), expected)
        factual_hits_total += f_h
        factual_exp_total  += len(expected)
        sem_hits2_total    += s_h
        print(f"{marker} F={f_h}/{len(expected)} S={s_h}/{len(expected)}  {q[:45]}")

    if missed_routes:
        print(f"\n  !! {len(missed_routes)} MISSED ROUTES: {missed_routes}")
    if wrong_indicator:
        print(f"  !! {len(wrong_indicator)} WRONG INDICATORS: {wrong_indicator}")
    if not missed_routes and not wrong_indicator:
        print(f"\n  All {len(SHOULD_ROUTE)} factual queries routed correctly.")
    print(f"  Factual hits@20 : {factual_hits_total}/{factual_exp_total} ({100*factual_hits_total/factual_exp_total:.0f}%)")
    print(f"  Semantic hits@20: {sem_hits2_total}/{factual_exp_total} ({100*sem_hits2_total/factual_exp_total:.0f}%)")
    print(f"  Factual gain    : +{factual_hits_total - sem_hits2_total} hits")

    # ── Group 3: Edge cases ───────────────────────────────────────────────
    print(f"\nGroup 3: Edge cases (should NOT route) — {len(EDGE_CASES_NO_ROUTE)} queries")
    edge_wrong = []
    for q, _ in EDGE_CASES_NO_ROUTE:
        route = detect_route(q)
        if route is not None:
            edge_wrong.append((q, route))
            print(f"  WRONG ROUTE  '{q}' → {route[0]}")
        else:
            print(f"  ok  '{q}'")

    if not edge_wrong:
        print(f"\n  All {len(EDGE_CASES_NO_ROUTE)} edge cases correctly not routed.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_failures = len(wrong_routes) + len(missed_routes) + len(wrong_indicator) + len(edge_wrong)
    if total_failures == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {total_failures} FAILURE(S) — see above")
    print(f"  Semantic fallback (Group 1):  {sem_hits_total}/{sem_exp_total} hits@20")
    print(f"  Factual routing (Group 2):    {factual_hits_total}/{factual_exp_total} hits@20")
    print(f"  vs Semantic on same queries:  {sem_hits2_total}/{factual_exp_total} hits@20")


if __name__ == "__main__":
    run()
