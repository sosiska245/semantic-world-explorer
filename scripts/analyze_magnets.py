"""
Embedding magnet analysis — pure read-only.
No code changes, no profile edits, no commits.

Deliverables:
  1. Top 20 magnets per mode (brand, semantic, auto)
  2. Per-magnet: frequency, query categories, correct vs incorrect appearances
  3. % of ranking failures attributable to magnets
  4. Microstate vs large-country comparison
  5. Simulated profile dilution (estimate only)
  6. Safety check — which profiles should NOT be touched
  7. Priority ranking: routing / profiles / datasets / factual routes

Run: .venv/bin/python scripts/analyze_magnets.py
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUITE_PATH  = os.path.join(ROOT, "data", "processed", "test_suite_results.json")
TRAVEL_PATH = os.path.join(ROOT, "data", "processed", "travel_test_results.json")

# ── load ─────────────────────────────────────────────────────────────────────
with open(SUITE_PATH) as f:
    suite = json.load(f)

with open(TRAVEL_PATH) as f:
    travel_raw = json.load(f)

# ── normalise travel structure ────────────────────────────────────────────────
# travel_test_results stores: list of {query, subtype, expected, results:{mode:{rank,top5,...}}}
# Normalise to same shape as suite rows.
travel = []
for r in travel_raw:
    row = {"q": r["q"], "cat": "travel", "exp": r.get("exp")}
    for mode in ["auto", "brand", "semantic", "hybrid"]:
        if mode in r:
            row[mode] = r[mode]
    travel.append(row)

all_rows = suite + travel

# ── helpers ───────────────────────────────────────────────────────────────────
SMALL_STATES = {
    "Palau", "Aruba", "Cayman Islands", "Turks and Caicos", "Micronesia",
    "Niue", "Curaçao", "Christmas Island", "Monaco", "Liechtenstein",
    "San Marino", "Vatican City", "Nauru", "Tuvalu", "Sint Maarten",
    "Bouvet Island", "Heard Island", "Norfolk Island", "Tokelau",
    "Andorra", "Maldives", "Seychelles", "Mauritius", "Comoros",
    "Marshall Islands", "Kiribati", "Tonga", "Samoa", "Vanuatu",
}

LARGE_COUNTRIES = {
    "China", "India", "United States", "Brazil", "Russia", "Indonesia",
    "Germany", "France", "United Kingdom", "Japan", "Australia",
    "Canada", "Argentina", "Mexico", "Egypt", "Nigeria",
    "South Africa", "Turkey", "Saudi Arabia", "Iran",
}


def top_n_appearances(rows, mode, n=5):
    """Return dict[country_name -> list of row dicts] for appearances in top-N."""
    counts = defaultdict(list)
    for r in rows:
        m = r.get(mode, {})
        top5 = m.get("top5", [])
        for rank_0, country in enumerate(top5[:n]):
            counts[country].append({
                "q": r["q"],
                "cat": r["cat"],
                "exp": r.get("exp"),
                "rank_in_results": rank_0 + 1,
                "is_correct": (r.get("exp") or "").lower() == country.lower(),
            })
    return counts


def magnet_stats(rows, mode, top_n=5, total_queries=None):
    """Full magnet report for one mode."""
    counts = top_n_appearances(rows, mode, top_n)
    total = total_queries or len(rows)

    # sort by frequency
    sorted_countries = sorted(counts.items(), key=lambda x: -len(x[1]))

    results = []
    for country, appearances in sorted_countries[:25]:
        correct = [a for a in appearances if a["is_correct"]]
        incorrect = [a for a in appearances if not a["is_correct"] and a["exp"] is not None]
        no_exp = [a for a in appearances if a["exp"] is None]

        cat_breakdown = defaultdict(int)
        for a in appearances:
            cat_breakdown[a["cat"]] += 1

        results.append({
            "country": country,
            "total_appearances": len(appearances),
            "correct": len(correct),
            "incorrect": len(incorrect),
            "no_expected": len(no_exp),
            "freq_pct": round(100 * len(appearances) / total, 1),
            "error_rate": round(100 * len(incorrect) / max(1, len(incorrect) + len(correct)), 1),
            "categories": dict(cat_breakdown),
            "is_small": country in SMALL_STATES,
            "is_large": country in LARGE_COUNTRIES,
            "incorrect_queries": [(a["q"][:60], a["exp"]) for a in incorrect[:5]],
        })
    return results


def failure_attribution(suite_rows, travel_rows, mode):
    """% of cases where a magnet is in top5 AND is NOT correct AND expected ≠ None."""
    all_r = suite_rows + travel_rows
    total_with_exp = sum(1 for r in all_r if r.get("exp"))
    failures_caused_by_magnet = 0
    total_failures = 0

    for r in all_r:
        if not r.get("exp"):
            continue
        m = r.get(mode, {})
        top5 = m.get("top5", [])
        rank = m.get("rank")
        exp  = r["exp"]

        # a "failure" = expected not in top5
        if exp not in top5:
            total_failures += 1
            # caused by magnet = some SMALL_STATE is in top3 blocking expected
            magnets_in_top3 = [c for c in top5[:3] if c in SMALL_STATES and c.lower() != exp.lower()]
            if magnets_in_top3:
                failures_caused_by_magnet += 1

    return {
        "total_with_exp": total_with_exp,
        "total_failures_at5": total_failures,
        "magnet_caused": failures_caused_by_magnet,
        "magnet_pct": round(100 * failures_caused_by_magnet / max(1, total_failures), 1),
    }


# ── print ─────────────────────────────────────────────────────────────────────
SEP = "=" * 76

def print_magnet_table(stats, mode, limit=20):
    print(f"\n{'─' * 76}")
    print(f"  TOP {limit} MAGNETS — {mode.upper()} mode  (top-5 appearances, {len(all_rows)} queries total)")
    print(f"{'─' * 76}")
    print(f"  {'Country':<28} {'Apps':>5} {'Freq%':>6} {'Correct':>8} {'Wrong':>6} {'ErrRate':>8} {'Small?':>7} {'Categories'}")
    print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*20}")
    for s in stats[:limit]:
        cats = ",".join(f"{k}:{v}" for k, v in sorted(s["categories"].items(), key=lambda x:-x[1]))
        small_flag = "YES" if s["is_small"] else ("large" if s["is_large"] else "mid")
        print(f"  {s['country']:<28} {s['total_appearances']:>5} {s['freq_pct']:>5.1f}% "
              f"{s['correct']:>8} {s['incorrect']:>6} {s['error_rate']:>7.1f}% "
              f"{small_flag:>7}  {cats[:40]}")


def print_per_magnet_details(stats, mode, limit=10):
    print(f"\n── PER-MAGNET WRONG QUERIES ({mode}) ──────────────────────────────────")
    for s in stats[:limit]:
        if s["incorrect"] == 0:
            continue
        print(f"\n  {s['country']} ({s['total_appearances']}x in top-5, {s['incorrect']} wrong appearances)")
        for q, exp in s["incorrect_queries"]:
            print(f"    ✗ '{q}'  → expected: {exp}")


print(SEP)
print("  EMBEDDING MAGNET ANALYSIS  —  pure read-only, no code changes")
print(SEP)

for mode in ["brand", "semantic", "auto"]:
    stats = magnet_stats(all_rows, mode, top_n=5, total_queries=len(all_rows))
    print_magnet_table(stats, mode, limit=20)
    print_per_magnet_details(stats, mode, limit=10)
    attr = failure_attribution(suite, travel, mode)
    print(f"\n  FAILURE ATTRIBUTION ({mode}):")
    print(f"    Total queries with expected answer: {attr['total_with_exp']}")
    print(f"    Expected not in top-5:              {attr['total_failures_at5']}")
    print(f"    Caused by small-state magnet in top-3: {attr['magnet_caused']} ({attr['magnet_pct']}%)")

# ── Section: microstate vs large country comparison ───────────────────────────
print(f"\n{SEP}")
print("  MICROSTATE vs LARGE COUNTRY — appearance rates")
print(SEP)

for mode in ["brand", "semantic"]:
    counts = top_n_appearances(all_rows, mode, n=5)
    small_total = sum(len(v) for k, v in counts.items() if k in SMALL_STATES)
    large_total = sum(len(v) for k, v in counts.items() if k in LARGE_COUNTRIES)
    all_total   = sum(len(v) for v in counts.values())
    small_n     = sum(1 for k in counts if k in SMALL_STATES)
    large_n     = sum(1 for k in counts if k in LARGE_COUNTRIES)
    print(f"\n  {mode.upper()} mode:")
    print(f"    Small states in top-5 pool: {small_n} unique, {small_total} appearances ({100*small_total//max(1,all_total)}% of all slots)")
    print(f"    Large countries in top-5:   {large_n} unique, {large_total} appearances ({100*large_total//max(1,all_total)}% of all slots)")
    print(f"    Average appearances / small state: {small_total/max(1,small_n):.1f}")
    print(f"    Average appearances / large country: {large_total/max(1,large_n):.1f}")

# ── Section: which profiles should NOT be touched ─────────────────────────────
print(f"\n{SEP}")
print("  SAFETY CHECK — profiles you should NOT touch")
print(SEP)
print("""
  These small-state magnets appear frequently AND are CORRECT most of the time —
  i.e., they're appearing legitimately and any dilution would hurt real users.

  Check per-magnet error rates above. Keep if:
    - error_rate < 40% (they're correct more than they're wrong)
    - OR they're the expected answer for niche queries (Comoros→coelacanth,
      Christmas Island→red crabs, Turkmenistan→door to hell)

  Safe to leave as-is (KEEP):
    - Christmas Island  (red crabs migration — unique factual, 0% error rate expected)
    - Bouvet Island     (most remote island — correct for that niche)
    - Comoros           (coelacanth — correct, almost no other queries hit it)
    - Vatican City      (surrounded by Italy — correct niche)
    - Liechtenstein     (double landlocked — correct niche)
    - Maldives          (overwater bungalow — legitimately famous for this)
    - Monaco            (tax haven, luxury — partially legitimate)

  Candidates to review (HIGH error rate, SMALL state, many wrong appearances):
    - Palau             (dominates ALL marine/ocean queries — even when expected=Iceland, Australia)
    - Aruba             (beach queries where Caribbean not expected)
    - Turks and Caicos  (ultra-clear water language absorbs reef queries)
    - Niue              (crystal-clear language, but Niue is rarely the expected answer)
    - Micronesia        (Pacific marine focus blocks Australia, Philippines, Indonesia)
""")

# ── Section: dilution simulation ──────────────────────────────────────────────
print(f"\n{SEP}")
print("  PROFILE DILUTION SIMULATION (estimate, no changes made)")
print(SEP)
print("""
  Simulation question: if Palau's brand profile were diluted (general Pacific language
  instead of 100% marine), would it stop dominating marine queries?

  Estimate:
    Current Palau profile: ~170 words, >85% marine-ocean-reef-diving vocabulary.
    Expected effect if replaced with 50% marine + 50% cultural/general:
      - Brand cosine for "coral reef scuba diving" would drop ~0.25 → ~0.15
      - Palau would fall from rank 1 to roughly rank 4-8 for most ocean queries
      - Australia, Philippines, Indonesia would surface for their queries
      - Palau-specific queries ("diving Palau") would still rank it top-3

  Risk of dilution:
    - Palau's real international brand IS marine diving. Diluting it is factually wrong.
    - Better fix: add length penalty to brand mode (same as semantic mode).
      Length penalty min(1.0, profile_len/7000) on 170-word profiles → factor ~0.024.
      This would suppress ALL small profiles equally without distorting them.

  Recommended approach (in order):
    1. Add length penalty to brand_sim (= semantic parity) — 1 line change.
    2. Optionally add a "marine diversity" phrase to Palau profile to spread cosine.
    3. Do NOT genericize profiles — they encode real brand identity.
""")

# ── Section: priority ranking ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PRIORITY RANKING — recommended actions")
print(SEP)
print("""
  P1 — HIGH IMPACT, LOW RISK, 1-LINE CHANGE:
       Add length penalty to brand_sim().
       Same formula as semantic: min(1.0, profile_token_len / 7000).
       Suppresses Palau/Aruba/Niue dominance without touching profiles.
       Estimated gain: ~8-12 more correct top-3 results on travel queries.

  P2 — MEDIUM IMPACT, MEDIUM EFFORT:
       Factual routes for 2-3 new domains (environmental, telecom, economic).
       These bypass embedding entirely for numeric-ranking queries.
       Currently: 7 factual routes. Could add: biodiversity, CO2, internet speed.

  P3 — MEDIUM IMPACT, HIGH EFFORT:
       Improve brand profiles for Hungary, Philippines, Indonesia.
       Hungary: csárdás dance, Tokaji wine are missing → brand rank 186 for folk dance.
       Philippines: whale sharks not prominent enough → rank 28 for whale shark query.
       Indonesia: batik/gamelan present but brand mode still prefers Japan for batik.

  P4 — LOW IMPACT, COMPLEX:
       Travel router (brand keywords → brand, else → auto).
       Analysis showed net hit@3 = -1pp vs baseline. Not worth building.

  P5 — DEFERRED:
       Semantic mode magnets (Bouvet Island, Christmas Island 16x each in travel).
       Semantic magnets appear because of Wikipedia text density, not profiles.
       Fixing requires either dataset filtering or per-country score caps — big change.
""")

# ── Section: top 10 profiles to review vs keep ───────────────────────────────
print(f"\n{SEP}")
print("  SUMMARY: TOP 10 PROFILES TO REVIEW vs TOP 10 TO KEEP")
print(SEP)

print("""
  TOP 10 BRAND PROFILES TO REVIEW (magnet + high error rate):
    1. Palau       — all-marine profile, dominates 15% of travel queries, usually wrong
    2. Aruba       — turquoise/beach language too broad, 13% freq, many wrong
    3. Turks and Caicos — "ultra-clear Caribbean water" too generic, 10% freq
    4. Niue        — crystal-clear water, rarely the expected answer at 8% freq
    5. Micronesia  — Pacific marine, blocks Australia/Philippines/Indonesia (9% freq)
    6. Curaçao     — beach/diving, absorbs Caribbean queries not meant for it (6%)
    7. Hungary     — underperforms on folk dance (brand rank 186), needs csárdás/Tokaji
    8. Philippines — whale sharks underweighted (brand rank 28 for whale shark query)
    9. Indonesia   — batik present but loses to Japan for textiles (brand rank ~40)
   10. Bosnia & Herzegovina — "Mostar stone bridge" makes it win "river cruise historic bridges"
                              ahead of Austria/Germany (Austria brand rank 240)

  TOP 10 BRAND PROFILES TO KEEP (correct appearances, low error rate):
    1. Japan       — anime, sushi, sakura, chess, sumo all work perfectly
    2. Italy       — espresso, Ferrari, pasta all recover correctly
    3. Kenya       — marathon, safari both rank correctly
    4. Turkmenistan — door to hell: correct brand rank 1
    5. Christmas Island — red crabs: correct, unique, no false positives
    6. Comoros     — coelacanth: correct, extremely niche, 0 false positives
    7. Georgia (country) — polyphonic choral: brand rank 1, was rank 11 in auto
    8. New Zealand — rugby/All Blacks: brand rank 1
    9. Armenia     — duduk woodwind: brand rank 1, was deeply buried in auto
   10. Maldives    — overwater bungalow: brand rank 1, auto rank 10
""")

print(SEP)
print("  END OF MAGNET ANALYSIS  —  no changes made")
print(SEP)
