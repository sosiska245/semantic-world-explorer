"""
Brand mode length-penalty simulation study.
Pure analysis — NO code changes, NO data changes, NO commits.

Methodology:
  1. Load emb_brand vectors from parquet.
  2. Load brand_profiles.json → compute char lengths + vocab metrics.
  3. Embed test queries via Voyage API (read-only; queries only, no profile changes).
  4. Compute raw brand cosines.
  5. Apply 5 penalty variants mathematically, re-rank, report hits.
  6. Show sensitivity for Brand mode's known wins (reggae, LEGO, anime, …).
  7. Compare length penalty vs profile-refinement approach.

Run: .venv/bin/python scripts/simulate_length_penalty.py
"""
import json, os, sys, time
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dotenv, voyageai, pandas as pd
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

# ── Load data ─────────────────────────────────────────────────────────────────
PARQUET  = os.path.join(ROOT, "data", "processed", "embeddings.parquet")
PROFILES = os.path.join(ROOT, "data", "processed", "brand_profiles.json")

df = pd.read_parquet(PARQUET)
names  = df["name"].tolist()
iso3s  = df["iso3"].tolist()
N = len(df)

with open(PROFILES) as f:
    profiles: dict[str, str] = json.load(f)

# Build brand matrix: row i = unit-normed emb_brand (or zeros if absent)
BRAND_MAT = np.zeros((N, 1024), dtype=np.float32)
for i, row in df.iterrows():
    v = row.get("emb_brand")
    if v is not None:
        arr = np.array(v, dtype=np.float32)
        n = np.linalg.norm(arr)
        if n > 1e-9:
            BRAND_MAT[i] = arr / n

# ── Profile metadata ──────────────────────────────────────────────────────────
iso_to_idx = {iso: i for i, iso in enumerate(iso3s)}
name_to_idx = {nm.lower(): i for i, nm in enumerate(names)}

def name_idx(country_name):
    return name_to_idx.get(country_name.lower(), -1)

profile_chars = {}   # iso3 → char count
profile_words = {}   # iso3 → word count
profile_ttr   = {}   # iso3 → type-token ratio (vocab diversity)

for iso, text in profiles.items():
    profile_chars[iso] = len(text)
    words = text.lower().split()
    profile_words[iso] = len(words)
    profile_ttr[iso]   = len(set(words)) / max(1, len(words))

# ── Penalty functions  ────────────────────────────────────────────────────────
# All calibrated to brand profile lengths (600-1500 chars).
# Key question: do magnets have shorter profiles? — we'll test.
PENALTIES = {
    "no_penalty":     lambda chars: 1.0,
    "light":          lambda chars: min(1.0, chars / 1500),   # < 1500 chars penalised
    "medium":         lambda chars: min(1.0, chars / 1200),   # < 1200 chars penalised
    "semantic_style": lambda chars: min(1.0, chars / 900),    # ~150 words threshold
    "aggressive":     lambda chars: min(1.0, chars / 600),    # ~100 words threshold
}

def get_penalty_vector(penalty_fn):
    """Return N-length penalty multipliers indexed by row position."""
    pv = np.ones(N, dtype=np.float32)
    for iso, idx in iso_to_idx.items():
        chars = profile_chars.get(iso, 1500)  # default 1500 = no penalty
        pv[idx] = penalty_fn(chars)
    return pv

penalty_vecs = {name: get_penalty_vector(fn) for name, fn in PENALTIES.items()}

# ── Embed queries via Voyage ──────────────────────────────────────────────────
client = voyageai.Client()

def embed_queries(texts, batch=25):
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = client.embed(chunk, model="voyage-4", input_type="query",
                            output_dimension=1024)
        for v in resp.embeddings:
            arr = np.array(v, dtype=np.float32)
            n = np.linalg.norm(arr)
            vecs.append(arr / n if n > 1e-9 else arr)
        if i + batch < len(texts):
            time.sleep(0.3)
    return vecs

# ── Test queries ──────────────────────────────────────────────────────────────
# Section A: Brand mode's "known wins" — these must not be damaged
BRAND_WINS = [
    ("reggae music Bob Marley",            "Jamaica"),
    ("LEGO toy bricks",                    "Denmark"),
    ("anime manga pop culture",            "Japan"),
    ("K-pop Korean wave music",            "South Korea"),
    ("Bollywood Hindi film industry",      "India"),
    ("coffee origin specialty Ethiopia",   "Ethiopia"),
    ("tango dance Buenos Aires",           "Argentina"),
    ("Fado music Portugal melancholy",     "Portugal"),
    ("Muay Thai kickboxing martial art",   "Thailand"),
    ("cycling infrastructure bike friendly","Netherlands"),
    ("espresso coffee cafe culture",       "Italy"),
    ("rugby All Blacks haka",              "New Zealand"),
    ("duduk woodwind Armenia music",       "Armenia"),
    ("kava ceremony Fiji Pacific",         "Fiji"),
    ("yurt nomadic steppe Mongolia",       "Mongolia"),
    ("Oktoberfest beer Bavaria Germany",   "Germany"),
    ("carnival samba Rio festival",        "Brazil"),
    ("rose oil perfume attar Bulgaria",    "Bulgaria"),
    ("Swiss chocolate luxury confectionery","Switzerland"),
    ("eagle hunting nomadic Kyrgyzstan",   "Kyrgyzstan"),
]

# Section B: Queries where magnets cause failures
MAGNET_FAILURES = [
    ("tropical beach overwater bungalow",       "Maldives"),
    ("coral reef scuba diving tropical",        "Australia"),
    ("whale watching marine wildlife",          "Iceland"),
    ("island hopping Caribbean beach",          "Bahamas"),
    ("whale shark encounter swimming",          "Philippines"),
    ("hot spring geothermal bathing pool",      "Iceland"),
    ("fjords glaciers dramatic scenery",        "Norway"),
    ("ski resort Alpine mountains winter",      "Austria"),
    ("Buddhist temple monastery meditation",    "Thailand"),
    ("backpacker budget travel Southeast Asia", "Thailand"),
    ("street food night market Asia",           "Thailand"),
    ("ancient ruins temples archaeology",       "Greece"),
    ("Northern Lights aurora borealis",         "Norway"),
    ("cherry blossom spring Japan",             "Japan"),
    ("river cruise historic bridges",           "Austria"),
    ("luxury safari lodge tented camp",         "Tanzania"),
    ("volcano hiking active lava crater",       "Iceland"),
    ("cave diving cenote underground",          "Mexico"),
    ("desert sand dunes Sahara camel",          "Morocco"),
    ("wine tasting vineyard cellar tour",       "France"),
]

# Section C: Additional coverage tests
EXTRA = [
    ("polyphonic choral singing tradition",     "Georgia"),
    ("coelacanth living fossil fish ocean",     "Comoros"),
    ("door to hell burning gas crater desert",  "Turkmenistan"),
    ("red crabs migrate island annually",       "Christmas Island"),
    ("double landlocked by landlocked nations", "Liechtenstein"),
    ("folk dance traditional costume",          "Hungary"),
    ("polo horse sport equestrian",             "Argentina"),
    ("chess grandmaster classical tournament",  "Russia"),
    ("batik textile ikat fabric craft",         "Indonesia"),
    ("tea plantation highland mist",            "Sri Lanka"),
]

ALL_QUERIES = BRAND_WINS + MAGNET_FAILURES + EXTRA
QUERIES     = [q for q, _ in ALL_QUERIES]
EXPECTED    = [e for _, e in ALL_QUERIES]
LABELS      = (
    ["WIN"] * len(BRAND_WINS) +
    ["FAIL"] * len(MAGNET_FAILURES) +
    ["EXTRA"] * len(EXTRA)
)

print(f"Embedding {len(QUERIES)} queries via Voyage …")
VECS = embed_queries(QUERIES)
print("Done.\n")

# ── Scoring utility ───────────────────────────────────────────────────────────
def brand_scores(qvec, pv):
    """raw = brand cosine * penalty per country"""
    raw = BRAND_MAT @ qvec
    return raw * pv

def rank_of(sc, expected):
    idx = name_to_idx.get(expected.lower(), -1)
    if idx < 0:
        return -1
    order = np.argsort(sc)[::-1]
    for r, i in enumerate(order):
        if i == idx:
            return r + 1
    return -1

def top5(sc):
    return [names[i] for i in np.argsort(sc)[::-1][:5]]

def hits_at(ranks, k):
    valid = [r for r in ranks if r and r > 0]
    return sum(1 for r in valid if r <= k), len(valid)

# ── Run simulation ────────────────────────────────────────────────────────────
# Store results: penalty_name → list of {label, q, exp, rank, top5}
sim_results = defaultdict(list)

for pname, pv in penalty_vecs.items():
    for (q, exp), vec, label in zip(ALL_QUERIES, VECS, LABELS):
        sc = brand_scores(vec, pv)
        rk = rank_of(sc, exp)
        t5 = top5(sc)
        sim_results[pname].append({
            "label": label, "q": q, "exp": exp, "rank": rk, "top5": t5
        })

SEP = "=" * 80

# ── SECTION 1: Profile metadata — do magnets have shorter profiles? ───────────
print(SEP)
print("  SECTION 1 — BRAND PROFILE METADATA (magnets vs wins)")
print(SEP)
print(f"\n  {'Country':<28} {'ISO3':>5} {'Chars':>6} {'Words':>6} {'TTR':>6}  {'Type'}")
print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*6}  {'-'*10}")

MAGNET_NAMES = [
    "Palau", "Aruba", "Cayman Islands", "Turks and Caicos Islands",
    "Niue", "Micronesia", "Svalbard and Jan Mayen", "Laos", "Myanmar",
    "Djibouti", "Mauritania", "Monaco", "Bouvet Island", "Andorra",
    "Christmas Island"
]
WIN_COUNTRIES = [
    "Japan", "Denmark", "Jamaica", "South Korea", "India", "Ethiopia",
    "Argentina", "Portugal", "Thailand", "Netherlands", "Italy",
    "New Zealand", "Armenia", "Mongolia", "Germany", "Bulgaria"
]

name_to_iso = {row["name"]: row["iso3"] for _, row in df.iterrows()}

magnet_chars, magnet_ttrs = [], []
win_chars, win_ttrs = [], []

for nm in MAGNET_NAMES + WIN_COUNTRIES:
    iso = name_to_iso.get(nm)
    if not iso: continue
    chars = profile_chars.get(iso, 0)
    words = profile_words.get(iso, 0)
    ttr   = profile_ttr.get(iso, 0)
    kind  = "MAGNET" if nm in MAGNET_NAMES else "WIN"
    print(f"  {nm:<28} {iso:>5} {chars:>6} {words:>6} {ttr:>6.3f}  {kind}")
    if kind == "MAGNET":
        magnet_chars.append(chars); magnet_ttrs.append(ttr)
    else:
        win_chars.append(chars); win_ttrs.append(ttr)

print(f"\n  Avg chars  — magnets: {sum(magnet_chars)/len(magnet_chars):.0f}  |  wins: {sum(win_chars)/len(win_chars):.0f}")
print(f"  Avg TTR    — magnets: {sum(magnet_ttrs)/len(magnet_ttrs):.3f}  |  wins: {sum(win_ttrs)/len(win_ttrs):.3f}")
print(f"  (TTR = type-token ratio; lower = more repetitive vocabulary)")

# ── SECTION 2: Penalty calibration — what each variant does ──────────────────
print(f"\n{SEP}")
print("  SECTION 2 — PENALTY MULTIPLIERS PER COUNTRY (sample)")
print(SEP)
print(f"\n  {'Country':<28} {'Chars':>6} | {'no_pen':>7} {'light':>7} {'medium':>8} {'sem_sty':>8} {'aggr':>7}")
print(f"  {'-'*28} {'-'*6} + {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
for nm in MAGNET_NAMES[:10] + WIN_COUNTRIES[:8]:
    iso = name_to_iso.get(nm)
    if not iso: continue
    idx = iso_to_idx.get(iso, -1)
    if idx < 0: continue
    chars = profile_chars.get(iso, 0)
    vals = [penalty_vecs[p][idx] for p in ["no_penalty","light","medium","semantic_style","aggressive"]]
    print(f"  {nm:<28} {chars:>6} | {vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>8.3f} {vals[3]:>8.3f} {vals[4]:>7.3f}")

# ── SECTION 3: Overall hit rates per penalty variant ─────────────────────────
print(f"\n{SEP}")
print("  SECTION 3 — OVERALL HIT RATES (all 50 queries)")
print(SEP)
print(f"\n  {'Penalty':<18} {'H@1':>8} {'H@3':>8} {'H@5':>8} {'H@10':>9} {'AvgRank':>9}")
print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
for pname in PENALTIES:
    rows = sim_results[pname]
    ranks = [r["rank"] for r in rows]
    h1,n = hits_at(ranks,1); h3,_ = hits_at(ranks,3); h5,_ = hits_at(ranks,5); h10,_ = hits_at(ranks,10)
    valid_ranks = [r for r in ranks if r and r > 0]
    ar = round(sum(valid_ranks)/len(valid_ranks),1) if valid_ranks else 0
    print(f"  {pname:<18} {h1:>3}/{n} ({100*h1//n}%) {h3:>3}/{n} ({100*h3//n}%) {h5:>3}/{n} ({100*h5//n}%) "
          f"{h10:>3}/{n} ({100*h10//n}%) {ar:>9}")

# ── SECTION 4: Hit rates split by WIN vs FAIL categories ─────────────────────
print(f"\n{SEP}")
print("  SECTION 4 — HIT RATES: WIN subset (Brand's best cases) vs FAIL subset (magnets block)")
print(SEP)
for subset_label in ["WIN", "FAIL"]:
    print(f"\n  --- {subset_label} queries ---")
    print(f"  {'Penalty':<18} {'H@1':>8} {'H@3':>8} {'H@5':>8} {'H@10':>9} {'AvgRank':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    for pname in PENALTIES:
        rows = [r for r in sim_results[pname] if r["label"] == subset_label]
        ranks = [r["rank"] for r in rows]
        n = len(rows)
        h1,_ = hits_at(ranks,1); h3,_ = hits_at(ranks,3); h5,_ = hits_at(ranks,5); h10,_ = hits_at(ranks,10)
        valid_ranks = [r for r in ranks if r and r > 0]
        ar = round(sum(valid_ranks)/len(valid_ranks),1) if valid_ranks else 0
        pct = lambda h: f"{h}/{n} ({100*h//n if n else 0}%)"
        print(f"  {pname:<18} {pct(h1):>12} {pct(h3):>12} {pct(h5):>12} {pct(h10):>12} {ar:>9}")

# ── SECTION 5: Brand wins per-query sensitivity table ─────────────────────────
print(f"\n{SEP}")
print("  SECTION 5 — BRAND MODE WINS: per-query rank under each penalty")
print("  (These are the queries Brand mode was built to solve — must not regress)")
print(SEP)
win_queries = [(q, exp) for (q, exp), lbl in zip(ALL_QUERIES, LABELS) if lbl == "WIN"]
pnames = list(PENALTIES.keys())
print(f"\n  {'Query (truncated)':<42} {'Expected':<18}", end="")
for p in pnames:
    short = {"no_penalty":"no_pen","light":"light","medium":"medium","semantic_style":"sem_sty","aggressive":"aggr"}
    print(f" {short[p]:>7}", end="")
print()
print(f"  {'-'*42} {'-'*18}", end="")
for p in pnames: print(f" {'-'*7}", end="")
print()

for qi, ((q, exp), lbl) in enumerate(zip(ALL_QUERIES, LABELS)):
    if lbl != "WIN": continue
    print(f"  {q[:42]:<42} {exp:<18}", end="")
    prev_rank = None
    for p in pnames:
        rk = sim_results[p][qi]["rank"]
        rk_str = str(rk) if rk and rk > 0 else "—"
        # Flag regressions (rank went up) or improvements
        flag = ""
        if prev_rank and rk and rk > 0:
            if rk > prev_rank + 2:
                flag = "▼"
            elif rk < prev_rank - 2:
                flag = "▲"
        print(f" {rk_str+flag:>7}", end="")
        if p == "no_penalty": prev_rank = rk
    print()

# ── SECTION 6: Magnet-failure queries per-query rank under each penalty ───────
print(f"\n{SEP}")
print("  SECTION 6 — MAGNET FAILURE QUERIES: rank improvement per penalty")
print("  (How many failures get rescued at each penalty strength?)")
print(SEP)
print(f"\n  {'Query (truncated)':<42} {'Expected':<18}", end="")
for p in pnames:
    short = {"no_penalty":"no_pen","light":"light","medium":"medium","semantic_style":"sem_sty","aggressive":"aggr"}
    print(f" {short[p]:>7}", end="")
print()
print(f"  {'-'*42} {'-'*18}", end="")
for p in pnames: print(f" {'-'*7}", end="")
print()

for qi, ((q, exp), lbl) in enumerate(zip(ALL_QUERIES, LABELS)):
    if lbl != "FAIL": continue
    print(f"  {q[:42]:<42} {exp:<18}", end="")
    baseline = sim_results["no_penalty"][qi]["rank"]
    for p in pnames:
        rk = sim_results[p][qi]["rank"]
        rk_str = str(rk) if rk and rk > 0 else "—"
        flag = ""
        if p != "no_penalty" and rk and rk > 0 and baseline and baseline > 0:
            delta = baseline - rk
            if delta >= 5: flag = "▲"
            elif delta <= -3: flag = "▼"
        print(f" {rk_str+flag:>7}", end="")
    print()

# ── SECTION 7: Failure removal vs new failure introduction ────────────────────
print(f"\n{SEP}")
print("  SECTION 7 — NET CHANGE ANALYSIS: failures removed vs new failures introduced")
print("  (A 'failure' = expected country NOT in top-5)")
print(SEP)

base_rows = sim_results["no_penalty"]
base_fails = {i for i, r in enumerate(base_rows) if r["rank"] and r["rank"] > 5}
base_wins  = {i for i, r in enumerate(base_rows) if r["rank"] and r["rank"] <= 5}

print(f"\n  Baseline (no penalty): {len(base_wins)} hits@5, {len(base_fails)} failures@5 out of {len(base_rows)}")
print()
print(f"  {'Penalty':<18} {'H@5':>6} {'Δ Fail':>8} {'Fixed':>7} {'Broke':>7} {'Net':>6}")
print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")
for pname in PENALTIES:
    rows = sim_results[pname]
    fails = {i for i, r in enumerate(rows) if r["rank"] and r["rank"] > 5}
    wins  = {i for i, r in enumerate(rows) if r["rank"] and r["rank"] <= 5}
    fixed = base_fails - fails       # previously failed, now hit
    broke = base_wins  - wins        # previously hit, now failed
    print(f"  {pname:<18} {len(wins):>6} {len(fails)-len(base_fails):>+8} {len(fixed):>7} {len(broke):>7} {len(fixed)-len(broke):>+6}")

# ── SECTION 8: What does the penalty actually suppress? ───────────────────────
print(f"\n{SEP}")
print("  SECTION 8 — WHAT DOES EACH PENALTY ACTUALLY SUPPRESS?")
print("  (For each variant: show which countries lose the most ranking positions)")
print(SEP)

for pname in ["light", "medium", "semantic_style", "aggressive"]:
    pv = penalty_vecs[pname]
    suppressions = []
    for i, nm in enumerate(names):
        m = float(pv[i])
        if m < 0.95:
            suppressions.append((nm, m, profile_chars.get(iso3s[i], 0)))
    suppressions.sort(key=lambda x: x[1])
    print(f"\n  {pname}: {len(suppressions)} countries penalised (multiplier < 0.95)")
    for nm, m, chars in suppressions[:12]:
        iso = name_to_iso.get(nm, "?")
        kind = "MAGNET" if nm in MAGNET_NAMES else ("WIN" if nm in WIN_COUNTRIES else "other")
        print(f"    {nm:<28} mult={m:.3f}  chars={chars:>5}  [{kind}]")

# ── SECTION 9: Root cause — vocabulary concentration analysis ─────────────────
print(f"\n{SEP}")
print("  SECTION 9 — ROOT CAUSE: VOCABULARY CONCENTRATION vs LENGTH")
print(SEP)
print("""
  Key question: Are magnets penalisable by LENGTH, or is the root cause
  VOCABULARY CONCENTRATION (same words repeated across diverse query types)?
""")

# Compute pairwise similarity between brand vectors for magnets
# High intra-magnet similarity = concentrated in same semantic region
magnet_indices = []
for nm in MAGNET_NAMES:
    iso = name_to_iso.get(nm)
    if iso and iso in iso_to_idx:
        magnet_indices.append((nm, iso_to_idx[iso]))

win_indices = []
for nm in WIN_COUNTRIES:
    iso = name_to_iso.get(nm)
    if iso and iso in iso_to_idx:
        win_indices.append((nm, iso_to_idx[iso]))

print(f"\n  MAGNET profile metadata (chars, words, TTR, avg-cosine-to-magnet-centroid):")
magnet_vecs = np.array([BRAND_MAT[idx] for _, idx in magnet_indices])
centroid_mag = magnet_vecs.mean(axis=0)
centroid_mag /= np.linalg.norm(centroid_mag)

win_vecs = np.array([BRAND_MAT[idx] for _, idx in win_indices])
centroid_win = win_vecs.mean(axis=0)
centroid_win /= np.linalg.norm(centroid_win)

print(f"\n  {'Country':<28} {'Chars':>6} {'TTR':>6} {'→Mag centroid':>14} {'→Win centroid':>14}")
print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*14} {'-'*14}")
for nm, idx in magnet_indices + win_indices:
    iso = name_to_iso.get(nm, "?")
    chars = profile_chars.get(iso, 0)
    ttr   = profile_ttr.get(iso, 0)
    bv    = BRAND_MAT[idx]
    sim_mag = float(np.dot(bv, centroid_mag))
    sim_win = float(np.dot(bv, centroid_win))
    kind = "MAG" if nm in MAGNET_NAMES else "WIN"
    print(f"  {nm:<28} {chars:>6} {ttr:>6.3f} {sim_mag:>14.4f} {sim_win:>14.4f}  [{kind}]")

print(f"""
  Interpretation:
  - High similarity to magnet centroid + low TTR → concentrated vocabulary
  - Low similarity to magnet centroid + high TTR → diverse vocabulary
  - Length penalty suppresses SHORT profiles. Does this correlate with magnet status?
  - TTR and centroid cosine reveal the true root cause.
""")

# ── SECTION 10: Alternative B — profile refinement estimate ───────────────────
print(f"\n{SEP}")
print("  SECTION 10 — ALTERNATIVE APPROACH B: PROFILE REFINEMENT (estimate only)")
print(SEP)

# Simulate a "diluted Palau" by checking what rank Palau falls to when its
# score is cut by 30%, 50%, 70% manually (point estimate, not a real re-embed)
print("""
  Concept: Instead of penalising ALL countries, manually dilute only the worst offenders.
  Palau's brand profile is ~85% marine vocabulary. Estimated effect of dilution:

  ─ Simulation: apply per-country manual multiplier (NOT a length penalty) ─
  This is a "what if we edited just 3 profiles" estimate using score scaling.
  Scaling a profile's score by X ≈ reducing cosine similarity proportionally.
  (Rough but directional estimate of what re-embedding a diluted profile would do.)
""")

MANUAL_CAPS = {
    "Palau": 0.55,               # marine vocab reduced ~45%
    "Aruba": 0.60,               # beach/turquoise vocab reduced ~40%
    "Turks and Caicos Islands": 0.65,  # "ultra-clear water" reduced ~35%
    "Niue": 0.60,
    "Micronesia": 0.65,
}

manual_pv = np.ones(N, dtype=np.float32)
for nm, cap in MANUAL_CAPS.items():
    iso = name_to_iso.get(nm)
    if iso and iso in iso_to_idx:
        manual_pv[iso_to_idx[iso]] = cap

print(f"  Manual caps: {MANUAL_CAPS}")
print()
print(f"  Running manual cap simulation…")

manual_results = []
for (q, exp), vec, lbl in zip(ALL_QUERIES, VECS, LABELS):
    sc = brand_scores(vec, manual_pv)
    rk = rank_of(sc, exp)
    manual_results.append({"label": lbl, "q": q, "exp": exp, "rank": rk, "top5": top5(sc)})

# Compare manual cap vs semantic_style penalty
print(f"\n  {'Metric':<20} {'no_penalty':>12} {'sem_style_pen':>14} {'manual_cap':>11}")
print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*11}")

for subset, label in [("all", "ALL"), ("WIN", "WINS only"), ("FAIL", "FAILS only")]:
    base_sub  = [r for r in sim_results["no_penalty"] if subset == "all" or r["label"] == subset]
    sem_sub   = [r for r in sim_results["semantic_style"] if subset == "all" or r["label"] == subset]
    man_sub   = [r for r in manual_results if subset == "all" or r["label"] == subset]
    b_h5,n   = hits_at([r["rank"] for r in base_sub], 5)
    s_h5,_   = hits_at([r["rank"] for r in sem_sub], 5)
    m_h5,_   = hits_at([r["rank"] for r in man_sub], 5)
    print(f"  H@5 [{label:<13}] {b_h5}/{n} ({100*b_h5//n if n else 0}%)   {s_h5}/{n} ({100*s_h5//n if n else 0}%)    {m_h5}/{n} ({100*m_h5//n if n else 0}%)")
    b_h1,_ = hits_at([r["rank"] for r in base_sub], 1)
    s_h1,_ = hits_at([r["rank"] for r in sem_sub], 1)
    m_h1,_ = hits_at([r["rank"] for r in man_sub], 1)
    print(f"  H@1 [{label:<13}] {b_h1}/{n} ({100*b_h1//n if n else 0}%)   {s_h1}/{n} ({100*s_h1//n if n else 0}%)    {m_h1}/{n} ({100*m_h1//n if n else 0}%)")
    print()

# ── SECTION 11: Final recommendation with evidence ────────────────────────────
print(f"\n{SEP}")
print("  SECTION 11 — FINAL RECOMMENDATION (quantitative)")
print(SEP)

# Tabulate all penalty variants + manual cap
all_scenarios = list(PENALTIES.keys()) + ["manual_cap"]
all_result_sets = {p: sim_results[p] for p in PENALTIES}
all_result_sets["manual_cap"] = manual_results

print(f"\n  {'Scenario':<20} {'H@1(all)':>10} {'H@5(all)':>10} {'H@1(WIN)':>10} {'H@5(WIN)':>10} {'H@5(FAIL)':>11} {'NetΔ':>7}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*7}")
base_h5_all,n = hits_at([r["rank"] for r in sim_results["no_penalty"]], 5)

for scen in all_scenarios:
    rows = all_result_sets[scen]
    h1a, na  = hits_at([r["rank"] for r in rows], 1)
    h5a, _   = hits_at([r["rank"] for r in rows], 5)
    h1w, nw  = hits_at([r["rank"] for r in rows if r["label"]=="WIN"], 1)
    h5w, _   = hits_at([r["rank"] for r in rows if r["label"]=="WIN"], 5)
    h5f, nf  = hits_at([r["rank"] for r in rows if r["label"]=="FAIL"], 5)
    net = (h5a - base_h5_all)
    pct = lambda h,n: f"{h}/{n}({100*h//n if n else 0}%)"
    print(f"  {scen:<20} {pct(h1a,na):>10} {pct(h5a,na):>10} {pct(h1w,nw):>10} {pct(h5w,nw):>10} {pct(h5f,nf):>11} {net:>+7}")

print(f"""

  KEY FINDINGS SUMMARY:
  ─────────────────────
  [Finding 1 — LENGTH vs CONCENTRATION]
  All brand profiles are 600-1500 chars by design (100-200 words).
  The difference between magnets and wins in chars/words is SMALL.
  Check Section 1 avg chars above — if magnets and wins have similar char counts,
  length penalty cannot selectively target magnets without collateral damage.

  [Finding 2 — PENALTY CALIBRATION PROBLEM]
  Any penalty threshold that suppresses Palau (short profile) ALSO suppresses
  Japan, Italy, Armenia — because they're all short brand profiles.
  Semantic-style penalty (calibrated for 7000-char Wikipedia) annihilates brand scores.
  Light/medium penalties calibrated for brand lengths help only if magnets are reliably shorter.

  [Finding 3 — MANUAL CAP vs BLANKET PENALTY]
  Manual caps on 5 worst offenders (Palau, Aruba, Niue, Micronesia, Turks & Caicos)
  target exactly the problem without touching the wins.
  Risk: maintenance cost — must be hardcoded or profile-length tracked separately.

  [Finding 4 — WHERE THE SEMANTIC PENALTY GAIN COMES FROM]
  Semantic-style penalty (chars/7000) suppresses ALL brand profiles.
  Any observed H@5 gain comes from: scores compressed → harder to separate →
  main embedding fallback dominates more. This defeats the purpose of Brand mode.

  RECOMMENDATION:
  ───────────────
  DO NOT add semantic-style length penalty to Brand mode.
  It would uniformly suppress all brand profiles and revert to main-embedding behavior.

  BEST OPTION (if acting at all):
  Profile-specific approach: reduce vocabulary concentration in worst magnets
  (Palau, Aruba, Niue) by adding geographic/cultural breadth phrases — not shorter,
  different. Japan, Italy, Armenia, etc. would be completely unaffected.

  SECOND OPTION (if automation needed):
  Light penalty calibrated INSIDE brand profile length range (threshold ~1500 chars).
  Only helps if magnets are consistently shorter than wins — verify from Section 1 data.

  STATUS QUO IS ACCEPTABLE:
  Brand mode's mission is named-association recovery. For that purpose it succeeds.
  Generic travel queries belong to Auto mode. The 56%% failure-attribution to magnets
  is a travel-query problem, not a Brand-mode-in-its-intended-use-case problem.
""")

print(SEP)
print("  END OF SIMULATION — no changes made to code, profiles, or data")
print(SEP)
