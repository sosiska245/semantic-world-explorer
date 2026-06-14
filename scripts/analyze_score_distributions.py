"""
Analyze cosine score distributions for strong vs weak queries.
Measures spread between rank-1/20/50 to evaluate whether current
0-100 normalization is misleading.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.data_loader import EMBEDDINGS, ENTITIES_DF
from src.multivec import apply_length_penalty, multi_sim
from src.similarity import embed_query

# ── query set ────────────────────────────────────────────────────────────────
# (label, query_text, category)
QUERIES = [
    # SPECIFIC — should produce large spread (clear winner)
    ("SPECIFIC", "sumo wrestling martial arts",          "Japan → clear"),
    ("SPECIFIC", "fjords glaciers Scandinavia",          "Norway/Iceland → clear"),
    ("SPECIFIC", "cacao cocoa bean plantation",          "Ivory Coast/Ghana → clear"),
    ("SPECIFIC", "Islam theocracy Sharia law",           "Iran/Saudi → clear"),
    ("SPECIFIC", "wine viticulture Bordeaux",            "France/Italy → clear"),
    ("SPECIFIC", "anime manga pop culture",              "Japan → clear"),
    ("SPECIFIC", "oil petroleum Gulf state",             "Saudi/UAE → clear"),
    ("SPECIFIC", "cricket bat wicket national sport",    "India/Pakistan → clear"),
    ("SPECIFIC", "tropical coral reef island",           "Maldives/Fiji → clear"),

    # MEDIUM — moderate spread expected
    ("MEDIUM",   "renewable energy solar wind power",    "Denmark/Norway"),
    ("MEDIUM",   "coffee culture cafe espresso",         "Ethiopia/Italy"),
    ("MEDIUM",   "startup venture capital tech hub",     "USA/Israel/Singapore"),
    ("MEDIUM",   "safari wildlife game reserve",         "Tanzania/Kenya"),
    ("MEDIUM",   "ancient civilisation archaeology",     "Greece/Egypt/Iraq"),

    # BROAD — expect compressed scores, weak results
    ("BROAD",    "developed country",                    "ambiguous"),
    ("BROAD",    "large population",                     "ambiguous"),
    ("BROAD",    "mountainous terrain",                  "ambiguous"),
    ("BROAD",    "coastal nation",                       "ambiguous"),
    ("BROAD",    "democracy",                            "ambiguous"),
    ("BROAD",    "technology",                           "ambiguous"),
]

N = len(ENTITIES_DF)

print(f"{'Cat':<9} {'Query':<40} {'sim#1':>7} {'sim#20':>7} {'sim#50':>7} {'Δ1-20':>7} {'Δ1-50':>7} {'pct20':>6} {'pct50':>6}")
print("-" * 105)

results = []

for cat, query, note in QUERIES:
    vec = embed_query(query)
    scores = multi_sim(query_vec=vec, alpha=0.30)
    scores = apply_length_penalty(scores)

    sorted_idx = np.argsort(scores)[::-1]
    s1  = float(scores[sorted_idx[0]])
    s20 = float(scores[sorted_idx[19]])
    s50 = float(scores[sorted_idx[49]])

    d1_20 = s1 - s20
    d1_50 = s1 - s50

    # current normalization: divide by top
    pct20 = int(round(s20 / s1 * 100))
    pct50 = int(round(s50 / s1 * 100))

    top5_names = [ENTITIES_DF.iloc[i]["name"] for i in sorted_idx[:5]]

    print(f"{cat:<9} {query:<40} {s1:>7.4f} {s20:>7.4f} {s50:>7.4f} {d1_20:>7.4f} {d1_50:>7.4f} {pct20:>5}% {pct50:>5}%")
    print(f"{'':9} note={note}")
    print(f"{'':9} top5: {', '.join(top5_names)}")
    print()

    results.append({"cat": cat, "query": query, "s1": s1, "s20": s20, "s50": s50,
                     "d1_20": d1_20, "d1_50": d1_50, "pct20": pct20, "pct50": pct50})

# ── aggregate by category ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("AGGREGATE STATS BY CATEGORY")
print("=" * 70)

for cat in ["SPECIFIC", "MEDIUM", "BROAD"]:
    subset = [r for r in results if r["cat"] == cat]
    avg = lambda k: np.mean([r[k] for r in subset])
    print(f"\n{cat} (n={len(subset)})")
    print(f"  avg sim#1  : {avg('s1'):.4f}")
    print(f"  avg sim#20 : {avg('s20'):.4f}")
    print(f"  avg sim#50 : {avg('s50'):.4f}")
    print(f"  avg Δ1-20  : {avg('d1_20'):.4f}   (bigger = more separation)")
    print(f"  avg Δ1-50  : {avg('d1_50'):.4f}")
    print(f"  avg pct@20 : {avg('pct20'):.1f}%  (current score shown for rank-20)")
    print(f"  avg pct@50 : {avg('pct50'):.1f}%  (current score shown for rank-50)")

# ── threshold analysis ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ABSOLUTE COSINE RANGES (all queries pooled)")
print("=" * 70)
all_s1  = [r["s1"]  for r in results]
all_s20 = [r["s20"] for r in results]
all_s50 = [r["s50"] for r in results]
print(f"  sim#1  range: {min(all_s1):.4f} – {max(all_s1):.4f}")
print(f"  sim#20 range: {min(all_s20):.4f} – {max(all_s20):.4f}")
print(f"  sim#50 range: {min(all_s50):.4f} – {max(all_s50):.4f}")
print()
print("  Queries where rank-20 scores ≥ 90% of rank-1 (misleadingly compressed):")
for r in results:
    if r["pct20"] >= 90:
        print(f"    [{r['cat']}] {r['query'][:45]:<45} rank20={r['pct20']}%  Δ1-20={r['d1_20']:.4f}")
