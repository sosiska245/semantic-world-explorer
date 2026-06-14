"""
Analyze test_suite_results.json — comprehensive quality report.
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "data", "processed", "test_suite_results.json")) as f:
    rows = json.load(f)

# ─── helpers ────────────────────────────────────────────────────────────────
def hits_at(rows_subset, mode, k):
    n, hit = 0, 0
    for r in rows_subset:
        if r.get("exp") is None: continue
        m = r.get(mode, {})
        rk = m.get("rank")
        if rk is None: continue
        n += 1
        if rk <= k: hit += 1
    return hit, n

def avg_rank(rows_subset, mode):
    ranks = [r[mode]["rank"] for r in rows_subset
             if r.get("exp") and r.get(mode, {}).get("rank") and r[mode]["rank"] > 0]
    return round(sum(ranks)/len(ranks), 1) if ranks else None

def conf_dist(rows_subset, mode):
    dist = {"HIGH": 0, "MED": 0, "LOW": 0}
    for r in rows_subset:
        c = r.get(mode, {}).get("conf")
        if c in dist: dist[c] += 1
    return dist

CATS = ["brand","travel","culture","factual","semantic","edge"]

print("=" * 72)
print("FULL TEST SUITE ANALYSIS  —  150 queries, 3 modes")
print("=" * 72)

# ── Section 1: Overall by mode ───────────────────────────────────────────────
print("\n── 1. OVERALL QUALITY BY MODE ──────────────────────────────────────")
print(f"{'Mode':<12} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>8} {'AvgRank':>9}")
print("-" * 55)
for mode in ["auto","brand","semantic"]:
    with_exp = [r for r in rows if r.get("exp") is not None]
    h1,n = hits_at(with_exp, mode, 1);  h3,_ = hits_at(with_exp, mode, 3)
    h5,_ = hits_at(with_exp, mode, 5);  h10,_ = hits_at(with_exp, mode, 10)
    ar = avg_rank(with_exp, mode)
    pct = lambda h: f"{h}/{n} ({100*h//n}%)" if n else "-"
    print(f"{mode:<12} {pct(h1):>12} {pct(h3):>12} {pct(h5):>12} {pct(h10):>12} {str(ar):>9}")

# ── Section 2: Best mode per category ────────────────────────────────────────
print("\n── 2. BEST MODE PER CATEGORY ────────────────────────────────────────")
print(f"{'Category':<12} {'N':>4} | {'Auto H@3':>10} {'Brand H@3':>10} {'Sem H@3':>10} | Winner")
print("-" * 65)
for cat in CATS:
    sub = [r for r in rows if r["cat"] == cat and r.get("exp")]
    if not sub: continue
    ah,n = hits_at(sub,"auto",3); bh,_ = hits_at(sub,"brand",3); sh,_ = hits_at(sub,"semantic",3)
    best = max([("auto",ah),("brand",bh),("semantic",sh)], key=lambda x: x[1])
    print(f"{cat:<12} {n:>4} | {ah:>4}/{n} ({100*ah//n if n else 0}%)  "
          f"{bh:>4}/{n} ({100*bh//n if n else 0}%)  "
          f"{sh:>4}/{n} ({100*sh//n if n else 0}%)  | {best[0].upper()}")

# ── Section 3: Brand mode — where it wins ────────────────────────────────────
print("\n── 3. BRAND MODE — WHERE IT CLEARLY WINS ───────────────────────────")
wins, losses, weird = [], [], []
for r in rows:
    if r["cat"] not in ("brand","culture","travel"): continue
    if not r.get("exp"): continue
    bm = r.get("brand",{}); am = r.get("auto",{})
    br = bm.get("rank"); ar = am.get("rank")
    if br is None or ar is None: continue
    bt5 = bm.get("top5",[]); at5 = am.get("top5",[])
    if br <= 3 and (ar is None or ar > 5):
        wins.append(r)
    elif br > 10 and (ar is not None and ar <= 5):
        losses.append(r)
    elif br > 15:
        weird.append(r)

print(f"  Brand clearly better (brand≤3, auto>5): {len(wins)} queries")
for r in wins:
    bm = r["brand"]; am = r["auto"]
    print(f"    ✓ '{r['q'][:50]}' → exp={r['exp']}, brand_rank={bm['rank']}, auto_rank={am.get('rank')}")
    print(f"        brand_top5: {bm['top5']}")

print(f"\n  Brand fails (brand>10, auto≤5): {len(losses)} queries")
for r in losses:
    bm = r["brand"]; am = r["auto"]
    print(f"    ✗ '{r['q'][:50]}' → exp={r['exp']}, brand_rank={bm['rank']}, auto_rank={am.get('rank')}")
    print(f"        brand_top5: {bm['top5']}")

print(f"\n  Brand strange results (rank>15): {len(weird)} queries")
for r in weird:
    bm = r["brand"]
    print(f"    ? '{r['q'][:50]}' → exp={r['exp']}, brand_rank={bm.get('rank')}")
    print(f"        brand_top5: {bm['top5']}")

# ── Section 4: Confidence badge vs result quality ─────────────────────────────
print("\n── 4. CONFIDENCE BADGE vs RESULT QUALITY ───────────────────────────")
for mode in ["auto","brand","semantic"]:
    print(f"\n  Mode: {mode}")
    for conf_lbl in ["HIGH","MED","LOW"]:
        sub = [r for r in rows if r.get(mode,{}).get("conf") == conf_lbl and r.get("exp")]
        h1,n = hits_at(sub, mode, 1)
        h5,_ = hits_at(sub, mode, 5)
        print(f"    {conf_lbl:<4}: {n:>3} queries  hit@1={h1}/{n} ({100*h1//n if n else 0}%)  hit@5={h5}/{n} ({100*h5//n if n else 0}%)")

# ── Section 5: Specific mode routing issues ───────────────────────────────────
print("\n── 5. AUTO ROUTING ANALYSIS ────────────────────────────────────────")
for r in rows:
    am = r.get("auto",{})
    tag = am.get("tag","")
    if "factual" in tag:
        pass  # expected for factual cat
    bm = r.get("brand",{})
    sm = r.get("semantic",{})
    # Cases where brand would massively help but auto doesn't use it
    if r["cat"] == "brand" and r.get("exp"):
        ar = am.get("rank"); br = bm.get("rank")
        if ar and br and ar > 20 and br <= 5:
            print(f"  AUTO-MISS brand: '{r['q'][:50]}' auto_rank={ar} brand_rank={br}")

factual_rows = [r for r in rows if r["cat"] == "factual"]
print(f"\n  Factual queries: {len(factual_rows)}")
for r in factual_rows:
    am = r.get("auto",{})
    tag = am.get("tag","")
    routed = "factual" in tag
    h1 = am.get("rank",999) <= 1
    marker = "✓" if routed else "~"
    exp_str = r.get("exp","?") or "?"
    top1 = am.get("top5",["?"])[0] if am.get("top5") else "?"
    print(f"    {marker} routed={routed:<5} '{r['q'][:40]}' → exp={exp_str[:20]}, top1={top1[:20]}")

# ── Section 6: Weak / underdescribed profiles ─────────────────────────────────
print("\n── 6. WEAK BRAND PROFILES (brand query, brand rank > 5) ─────────────")
brand_rows = [r for r in rows if r["cat"] == "brand"]
for r in brand_rows:
    bm = r.get("brand",{})
    rk = bm.get("rank")
    if rk and rk > 5:
        print(f"  WEAK [{rk:>3}] '{r['q'][:55]}' → exp={r.get('exp')}")
        print(f"         brand_top5: {bm.get('top5',[][:5])}")

# ── Section 7: Edge case behavior ─────────────────────────────────────────────
print("\n── 7. EDGE CASE BEHAVIOR ─────────────────────────────────────────────")
edge_rows = [r for r in rows if r["cat"] == "edge"]
for r in edge_rows:
    am = r.get("auto",{}); bm = r.get("brand",{})
    exp = r.get("exp")
    ar  = am.get("rank"); br  = bm.get("rank")
    at5 = am.get("top5",[]); bt5 = bm.get("top5",[])
    ac  = am.get("conf","?"); ad  = am.get("delta",0)
    ok_auto  = (ar and ar <= 5) if exp else True
    ok_brand = (br and br <= 5) if exp else True
    print(f"  '{r['q'][:50]}'")
    if exp:
        print(f"    exp={exp}  auto_rank={ar} brand_rank={br}")
    print(f"    auto_top5: {at5}  conf={ac}({ad})")

# ── Section 8: Map score distribution concerns ───────────────────────────────
print("\n── 8. SCORE DISTRIBUTION / MAP COLORING ────────────────────────────")
print("  Checking semantic delta distribution across query types …")
for cat in ["brand","travel","culture","semantic"]:
    sub = [r for r in rows if r["cat"] == cat]
    deltas = [r.get("semantic",{}).get("delta",0) for r in sub if r.get("semantic")]
    if deltas:
        avg_d = sum(deltas)/len(deltas)
        min_d = min(deltas); max_d = max(deltas)
        low_ct = sum(1 for d in deltas if d < 0.05)
        print(f"  {cat:<10}: avg_delta={avg_d:.3f} min={min_d:.3f} max={max_d:.3f} LOW_conf={low_ct}/{len(deltas)}")

# ── Section 9: Countries underdescribed / missing expected hits ───────────────
print("\n── 9. COUNTRY / PROFILE COVERAGE GAPS ──────────────────────────────")
miss_countries = {}
for r in rows:
    if not r.get("exp"): continue
    bm = r.get("brand",{})
    br = bm.get("rank")
    if br and br > 10:
        c = r["exp"]
        miss_countries[c] = miss_countries.get(c, 0) + 1

if miss_countries:
    for c, cnt in sorted(miss_countries.items(), key=lambda x: -x[1]):
        print(f"  {c}: {cnt} queries where brand rank > 10")

print("\n── 10. SUMMARY COUNTS ───────────────────────────────────────────────")
total = len(rows)
with_exp = [r for r in rows if r.get("exp")]
print(f"  Total queries: {total}")
print(f"  Queries with expected country: {len(with_exp)}")
for mode in ["auto","brand","semantic"]:
    h1,n = hits_at(with_exp, mode, 1)
    h5,_ = hits_at(with_exp, mode, 5)
    h10,_ = hits_at(with_exp, mode, 10)
    print(f"  {mode:<10}: hit@1={h1}/{n}={100*h1//n if n else 0}%  hit@5={h5}/{n}={100*h5//n if n else 0}%  hit@10={h10}/{n}={100*h10//n if n else 0}%")
