"""
Travel Query Analysis — Brand vs Auto vs Semantic vs Hybrid
Analysis-only: no code changes, no routing changes.

Run from project root: .venv/bin/python scripts/analyze_travel_queries.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dotenv; dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from src.data_loader import ENTITIES_DF, N_ENTITIES
from src.similarity import embed_query
from src.brand_sim import brand_sim
from src.multivec import apply_length_penalty, multi_sim
from src.bm25 import blend as bm25_blend
from src.domain_sim import domain_sim
from src.factual_router import detect_route, factual_scores

iso3  = ENTITIES_DF["iso3"].tolist()
names = ENTITIES_DF["name"].tolist()

HYBRID_TERMS = {
    "wheat","barley","cocoa","cacao","viticulture","paddy","plantation",
    "livestock","coffee production","wine production","tea production",
    "fishing industry","seafood export","rice cultivation",
}

def score_mode(mode, text, vec):
    if mode == "brand":
        return brand_sim(vec), "brand"
    if mode == "semantic":
        s = multi_sim(query_vec=vec, alpha=0.30)
        return apply_length_penalty(s), "semantic"
    if mode == "hybrid":
        s = multi_sim(query_vec=vec, alpha=0.30)
        s = apply_length_penalty(s)
        return bm25_blend(s, text, alpha=0.08), "hybrid"
    if mode == "domain":
        s = domain_sim(vec, alpha=0.4)
        return apply_length_penalty(s), "domain"
    # auto
    route = detect_route(text)
    if route is not None:
        ind, asc, label = route
        return factual_scores(ind, iso3, asc), f"factual:{ind}"
    s = multi_sim(query_vec=vec, alpha=0.30)
    s = apply_length_penalty(s)
    if any(t in text.lower() for t in HYBRID_TERMS):
        return bm25_blend(s, text, alpha=0.08), "auto:hybrid"
    return s, "auto:semantic"

def top_n(sc, n=5):
    return [names[i] for i in np.argsort(sc)[::-1][:n]]

def rank_of(sc, exp):
    if not exp: return None
    for r, i in enumerate(np.argsort(sc)[::-1]):
        if names[i].lower() == exp.lower(): return r + 1
    return -1

def conf(sc):
    s = sorted(sc, reverse=True)
    d = float(s[0] - s[19]) if len(s) >= 20 else 0.0
    return ("HIGH" if d > 0.10 else "MED" if d > 0.05 else "LOW"), round(d, 4)

# ─── 110 TRAVEL QUERIES ──────────────────────────────────────────────────────
# (query, subtype, expected_country, notes)
TRAVEL_QUERIES = [
    # BEACH DESTINATIONS (10)
    ("white sand beach turquoise water tropical",   "beach",   "Maldives",       "classic paradise beach"),
    ("pink sand beach Caribbean island",            "beach",   "Bahamas",        "Harbour Island pink sand"),
    ("black sand volcanic beach",                   "beach",   "Iceland",        "Reynisfjara"),
    ("long empty white beach Southeast Asia",       "beach",   "Philippines",    "El Nido / Palawan"),
    ("surf beach waves Atlantic coast",             "beach",   "Portugal",       "Algarve / Nazaré"),
    ("beach party full moon",                       "beach",   "Thailand",       "Koh Phangan"),
    ("hidden gem secluded cove Mediterranean",      "beach",   "Greece",         "lesser-known Greek islands"),
    ("beach resort all inclusive Caribbean",        "beach",   "Jamaica",        "Montego Bay etc"),
    ("pristine beach coral reef island",            "beach",   "Maldives",       ""),
    ("beach camping wild coastline",                "beach",   "New Zealand",    "Abel Tasman etc"),

    # DIVING / MARINE LIFE (10)
    ("scuba diving coral reef tropical water",      "diving",  "Australia",      "Great Barrier Reef"),
    ("whale shark snorkeling swimming",             "diving",  "Philippines",    "Donsol / Oslob"),
    ("manta ray dive open ocean",                   "diving",  "Maldives",       "South Ari Atoll"),
    ("wreck diving WWII shipwreck",                 "diving",  "Palau",          "Palau wrecks"),
    ("hammerhead shark diving archipelago",         "diving",  "Ecuador",        "Galapagos"),
    ("bioluminescent bay kayak swim",               "diving",  "Puerto Rico",    "Mosquito Bay Vieques"),
    ("cenote cave diving underground",              "diving",  "Mexico",         "Yucatan cenotes"),
    ("sea turtle nesting beach diving",             "diving",  "Costa Rica",     "Tortuguero"),
    ("night diving reef phosphorescence",           "diving",  "Maldives",       ""),
    ("drift diving strong current pristine reef",   "diving",  "Palau",          "Blue Corner"),

    # HIKING / MOUNTAINS (10)
    ("trek high altitude mountain base camp",       "hiking",  "Nepal",          "Everest Base Camp"),
    ("volcanic crater hike active lava",            "hiking",  "Iceland",        ""),
    ("fjord hike viewpoint panorama",               "hiking",  "Norway",         "Trolltunga Preikestolen"),
    ("rainforest jungle trek wildlife",             "hiking",  "Costa Rica",     ""),
    ("high altitude plateau trekking Andes",        "hiking",  "Peru",           "Inca Trail"),
    ("glacier hike ice walk crampons",              "hiking",  "Iceland",        "Vatnajokull"),
    ("Alps long-distance hiking trail",             "hiking",  "Switzerland",    ""),
    ("multi-day bush walk coastal",                 "hiking",  "New Zealand",    "Milford Track"),
    ("kilimanjaro climb summit Africa",             "hiking",  "Tanzania",       ""),
    ("Camino de Santiago pilgrimage walk",          "hiking",  "Spain",          ""),

    # CITY BREAKS (10)
    ("romantic weekend city break Europe",          "city",    "France",         "Paris"),
    ("art museum gallery city culture",             "city",    "Italy",          "Florence / Rome"),
    ("nightlife bar club city break",               "city",    "Germany",        "Berlin"),
    ("architecture design modernist city",          "city",    "Spain",          "Barcelona Gaudi"),
    ("canal waterways boat city",                   "city",    "Netherlands",    "Amsterdam"),
    ("historic old town cobblestone streets",       "city",    "Czech Republic", "Prague"),
    ("street food market city weekend",             "city",    "Thailand",       "Bangkok Chatuchak"),
    ("cosmopolitan skyline harbour city",           "city",    "Australia",      "Sydney"),
    ("cultural heritage mosque palace city",        "city",    "Morocco",        "Marrakesh"),
    ("Christmas market winter city break",          "city",    "Germany",        ""),

    # FOOD TRAVEL (10)
    ("gastronomic destination Michelin starred",    "food",    "France",         ""),
    ("street food hawker stalls authentic",         "food",    "Singapore",      ""),
    ("pasta wine Italian food pilgrimage",          "food",    "Italy",          ""),
    ("spicy cuisine chilli pepper hot",             "food",    "Mexico",         ""),
    ("fresh seafood coastal fishing village",       "food",    "Portugal",       ""),
    ("dim sum dumpling brunch culture",             "food",    "Hong Kong",      ""),
    ("sushi omakase chef experience",               "food",    "Japan",          ""),
    ("barbecue smoked meat grilling",               "food",    "United States",  "Texas BBQ"),
    ("vegetarian vegan spiritual food",             "food",    "India",          ""),
    ("ceviche raw fish lime citrus",                "food",    "Peru",           ""),

    # COFFEE / WINE TRAVEL (8)
    ("coffee origin farm tour bean single",         "coffee",  "Ethiopia",       ""),
    ("espresso coffee culture café sit",            "coffee",  "Italy",          ""),
    ("third wave specialty coffee scene",           "coffee",  "Australia",      "Melbourne"),
    ("wine region vineyard cellar door",            "wine",    "France",         "Bordeaux"),
    ("wine tour Tuscany rolling hills",             "wine",    "Italy",          "Chianti"),
    ("New World wine country tour",                 "wine",    "New Zealand",    "Marlborough"),
    ("champagne cellar sparkling wine",             "wine",    "France",         "Reims"),
    ("sake brewery rice wine tour",                 "wine",    "Japan",          ""),

    # CULTURAL EXPERIENCES (10)
    ("ancient temple ruins sunrise visit",          "cultural","Cambodia",       "Angkor Wat"),
    ("traditional ceremony ritual indigenous",      "cultural","Papua New Guinea","sing-sing"),
    ("mask festival cultural performance",          "cultural","Japan",          ""),
    ("muezzin call to prayer mosque quarter",       "cultural","Morocco",        ""),
    ("medieval walled city fortress heritage",      "cultural","Croatia",        "Dubrovnik"),
    ("Incan ruins stone terraces altitude",         "cultural","Peru",           "Machu Picchu"),
    ("Aboriginal art dreamtime culture",            "cultural","Australia",      ""),
    ("classical dance costume performance",         "cultural","Thailand",       "Khon dance"),
    ("royal palace ceremony changing guard",        "cultural","United Kingdom", ""),
    ("geisha tea house Japanese tradition",         "cultural","Japan",          "Kyoto"),

    # FAMOUS LANDMARKS (10)
    ("Eiffel Tower Paris iconic skyline",           "landmark","France",         ""),
    ("Great Wall historical monument",              "landmark","China",          ""),
    ("Colosseum Roman amphitheatre",               "landmark","Italy",          ""),
    ("Taj Mahal marble mausoleum",                  "landmark","India",          ""),
    ("Machu Picchu lost Inca city altitude",        "landmark","Peru",           ""),
    ("Pyramids of Giza sphinx desert",             "landmark","Egypt",          ""),
    ("Petra rose-red Nabataean city",               "landmark","Jordan",         ""),
    ("Angkor Wat temple sunrise Cambodia",          "landmark","Cambodia",       ""),
    ("Hagia Sophia Istanbul Ottoman",               "landmark","Turkey",         ""),
    ("Sagrada Familia Gaudi Barcelona",             "landmark","Spain",          ""),

    # WILDLIFE (10)
    ("Big Five safari lion elephant leopard",       "wildlife","Kenya",          "Masai Mara"),
    ("mountain gorilla trekking permit",            "wildlife","Rwanda",         ""),
    ("polar bear Arctic wildlife",                  "wildlife","Norway",         "Svalbard"),
    ("Galápagos marine iguana endemic wildlife",    "wildlife","Ecuador",        ""),
    ("orangutan rainforest Borneo",                 "wildlife","Malaysia",       ""),
    ("giant tortoise island endemic",              "wildlife","Ecuador",        "Galapagos"),
    ("jaguar pantanal wetland",                     "wildlife","Brazil",         ""),
    ("penguin colony subantarctic island",          "wildlife","Argentina",      "Falklands or Patagonia"),
    ("whale watching humpback breach",              "wildlife","Iceland",        ""),
    ("bird watching migration flamingo",            "wildlife","Kenya",          "Lake Nakuru"),

    # ADVENTURE TRAVEL (8)
    ("bungee jumping extreme sport",                "adventure","New Zealand",   ""),
    ("white water rafting river grade",             "adventure","Uganda",        "Nile source"),
    ("paragliding hang gliding mountain launch",    "adventure","Switzerland",   "Interlaken"),
    ("sand dune surfing desert sandboard",          "adventure","Namibia",       "Swakopmund"),
    ("skydiving scenic jump landscape",             "adventure","New Zealand",   ""),
    ("zip line canopy jungle adventure",            "adventure","Costa Rica",    ""),
    ("motorbike road trip mountain pass",           "adventure","Vietnam",       "Ha Giang"),
    ("deep sea fishing ocean charter",              "adventure","Maldives",      ""),

    # LUXURY TRAVEL (8)
    ("overwater villa private pool infinity",       "luxury",  "Maldives",       ""),
    ("private island exclusive resort",             "luxury",  "Maldives",       ""),
    ("butler service safari lodge",                 "luxury",  "South Africa",   ""),
    ("yacht charter Mediterranean island hop",      "luxury",  "Greece",         ""),
    ("six-star hotel iconic city",                  "luxury",  "United Arab Emirates","Burj Al Arab"),
    ("ski chalet luxury mountain resort",           "luxury",  "Switzerland",    "Verbier Gstaad"),
    ("wine and dine gourmet countryside",           "luxury",  "France",         ""),
    ("spa thermal wellness retreat",                "luxury",  "Iceland",        "Blue Lagoon"),

    # BACKPACKING (8)
    ("backpacker hostel cheap travel Asia",         "backpacking","Thailand",    ""),
    ("banana pancake trail Southeast Asia",         "backpacking","Thailand",    ""),
    ("train pass interrail budget Europe",          "backpacking","Germany",     ""),
    ("Inca trail independent trek",                 "backpacking","Peru",        ""),
    ("night bus overland budget journey",           "backpacking","Vietnam",     ""),
    ("island hopping cheap boat ferry",             "backpacking","Philippines", ""),
    ("couch surfing local host travel",             "backpacking","Germany",     "origin of Couchsurfing"),
    ("budget food market local eat cheap",          "backpacking","India",       ""),

    # ISLANDS (8)
    ("remote island off-grid escape",               "islands", "Maldives",       ""),
    ("volcanic tropical island lush green",         "islands", "Indonesia",      "Bali"),
    ("coral atoll lagoon Pacific paradise",         "islands", "French Polynesia","Bora Bora"),
    ("Greek island hopping whitewashed village",    "islands", "Greece",         "Cyclades"),
    ("Caribbean rum beach island",                  "islands", "Barbados",       ""),
    ("Nordic island fjord Arctic",                  "islands", "Iceland",        ""),
    ("Polynesian island culture canoe",             "islands", "Samoa",          ""),
    ("island nature reserve biodiversity",          "islands", "Madagascar",     ""),

    # WINTER TRAVEL (6)
    ("Northern Lights aurora hunting winter",       "winter",  "Norway",         ""),
    ("dog sledding husky Arctic winter",            "winter",  "Finland",        "Lapland"),
    ("skiing world-class piste resort",             "winter",  "Austria",        ""),
    ("ice hotel snow igloo sleep",                  "winter",  "Sweden",         "Icehotel Jukkasjärvi"),
    ("snowmobile glacier winter adventure",         "winter",  "Iceland",        ""),
    ("winter carnival snow festival",               "winter",  "Canada",         "Quebec City"),

    # FESTIVALS (6)
    ("music festival outdoor camping",              "festival","United Kingdom", "Glastonbury"),
    ("carnival Rio parade costume feathers",        "festival","Brazil",         ""),
    ("Oktoberfest beer tent Munich",                "festival","Germany",        ""),
    ("Diwali fireworks lights celebration",         "festival","India",          ""),
    ("cherry blossom hanami festival",              "festival","Japan",          ""),
    ("lantern festival night sky glow",             "festival","Thailand",       "Yi Peng Chiang Mai"),

    # ROAD TRIPS (6)
    ("scenic coastal road trip ocean view",         "roadtrip","New Zealand",    ""),
    ("Route 66 classic American road trip",         "roadtrip","United States",  ""),
    ("mountain pass hairpin alpine drive",          "roadtrip","Switzerland",    ""),
    ("wine country touring drive",                  "roadtrip","Australia",      "Barossa Valley"),
    ("safari self-drive game park",                 "roadtrip","Botswana",       ""),
    ("ring road circumnavigate island",             "roadtrip","Iceland",        ""),
]

MODES = ["auto", "brand", "semantic", "hybrid"]

print(f"Running {len(TRAVEL_QUERIES)} travel queries × {len(MODES)} modes …")
rows = []
for i, (query, subtype, expected, note) in enumerate(TRAVEL_QUERIES):
    vec = embed_query(query)
    row = {"q": query, "sub": subtype, "exp": expected, "note": note}
    for mode in MODES:
        sc, tag = score_mode(mode, query, vec)
        cl, cd  = conf(sc.tolist())
        t5      = top_n(sc, 5)
        rk      = rank_of(sc, expected)
        row[mode] = {"tag": tag, "top5": t5, "conf": cl, "delta": cd, "rank": rk}
    rows.append(row)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(TRAVEL_QUERIES)}")

print(f"Done: {len(rows)} rows")

def to_py(o):
    if isinstance(o, dict):  return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list):  return [to_py(x) for x in o]
    if isinstance(o, np.floating):  return float(o)
    if isinstance(o, np.integer):   return int(o)
    return o

rows = to_py(rows)
out = os.path.join(ROOT, "data", "processed", "travel_test_results.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Saved → {out}")

# ─── ANALYSIS ────────────────────────────────────────────────────────────────

def hits_at(subset, mode, k):
    n, hit = 0, 0
    for r in subset:
        if not r.get("exp"): continue
        rk = r.get(mode, {}).get("rank")
        if rk is None: continue
        n += 1
        if rk <= k: hit += 1
    return hit, n

def best_mode_for(r):
    """Return mode with lowest rank. Ties → first in MODES list."""
    best, best_rank = None, 9999
    for m in MODES:
        rk = r.get(m, {}).get("rank")
        if rk and rk < best_rank:
            best_rank = rk
            best = m
    return best, best_rank

SUBTYPES = [
    "beach","diving","hiking","city","food","coffee","wine",
    "cultural","landmark","wildlife","adventure","luxury",
    "backpacking","islands","winter","festival","roadtrip"
]

print("\n" + "=" * 76)
print("TRAVEL QUERY ANALYSIS REPORT")
print("=" * 76)

# ── Table 1: Hit@3 per mode per subtype ──────────────────────────────────────
print("\n── TABLE 1: Hit@3 by subtype × mode ──────────────────────────────────")
print(f"{'Subtype':<14} {'N':>3} | {'Auto':>8} {'Brand':>8} {'Semantic':>10} {'Hybrid':>8} | Best")
print("-" * 72)
subtype_winners = {}
for sub in SUBTYPES:
    sub_rows = [r for r in rows if r["sub"] == sub]
    if not sub_rows: continue
    n = len(sub_rows)
    scores = {}
    for m in MODES:
        h, nn = hits_at(sub_rows, m, 3)
        scores[m] = (h, nn)
    best_m = max(scores, key=lambda m: scores[m][0])
    subtype_winners[sub] = best_m
    def fmt(m): h,nn=scores[m]; return f"{h}/{nn}({100*h//nn if nn else 0}%)"
    print(f"{sub:<14} {n:>3} | {fmt('auto'):>9} {fmt('brand'):>9} {fmt('semantic'):>11} {fmt('hybrid'):>9} | {best_m.upper()}")

# ── Table 2: Hit@1 per subtype ────────────────────────────────────────────────
print("\n── TABLE 2: Hit@1 (exact top-1) by subtype × mode ───────────────────")
print(f"{'Subtype':<14} {'N':>3} | {'Auto':>8} {'Brand':>8} {'Semantic':>10} {'Hybrid':>8}")
print("-" * 66)
for sub in SUBTYPES:
    sub_rows = [r for r in rows if r["sub"] == sub]
    if not sub_rows: continue
    n = len(sub_rows)
    def fmt1(m):
        h,nn = hits_at(sub_rows, m, 1)
        return f"{h}/{nn}({100*h//nn if nn else 0}%)"
    print(f"{sub:<14} {n:>3} | {fmt1('auto'):>9} {fmt1('brand'):>9} {fmt1('semantic'):>11} {fmt1('hybrid'):>9}")

# ── Overall by mode ──────────────────────────────────────────────────────────
print("\n── OVERALL ACROSS ALL 110 TRAVEL QUERIES ─────────────────────────────")
print(f"{'Mode':<12} {'Hit@1':>9} {'Hit@3':>9} {'Hit@5':>9} {'Hit@10':>10}")
for m in MODES:
    h1,n=hits_at(rows,m,1); h3,_=hits_at(rows,m,3); h5,_=hits_at(rows,m,5); h10,_=hits_at(rows,m,10)
    print(f"{m:<12} {h1}/{n}({100*h1//n}%) {h3}/{n}({100*h3//n}%) {h5}/{n}({100*h5//n}%) {h10}/{n}({100*h10//n}%)")

# ── Per-query breakdown: who wins ────────────────────────────────────────────
print("\n── PER-QUERY WINNER BREAKDOWN ────────────────────────────────────────")
mode_win_counts = {m: 0 for m in MODES}
ties = 0
for r in rows:
    if not r.get("exp"): continue
    best, _ = best_mode_for(r)
    if best: mode_win_counts[best] += 1
print("  Best mode per query (lowest rank on expected country):")
for m in MODES:
    print(f"    {m:<12}: {mode_win_counts[m]} queries")

# ── Brand vs Semantic head-to-head ───────────────────────────────────────────
print("\n── BRAND vs SEMANTIC HEAD-TO-HEAD (all 110) ──────────────────────────")
brand_wins = sem_wins = tie_count = 0
for r in rows:
    if not r.get("exp"): continue
    br = r.get("brand",{}).get("rank"); sr = r.get("semantic",{}).get("rank")
    if br is None or sr is None: continue
    if br < sr: brand_wins += 1
    elif sr < br: sem_wins += 1
    else: tie_count += 1
print(f"  Brand wins: {brand_wins} | Semantic wins: {sem_wins} | Ties: {tie_count}")

# ── Brand clearly better (brand≤3, semantic>5) ───────────────────────────────
print("\n── BRAND CLEAR WINS (brand≤3, semantic>5) ────────────────────────────")
clear_brand = [(r, r["brand"]["rank"], r["semantic"]["rank"])
               for r in rows if r.get("exp")
               and r.get("brand",{}).get("rank","x") and r.get("semantic",{}).get("rank","x")
               and isinstance(r["brand"].get("rank"), int)
               and isinstance(r["semantic"].get("rank"), int)
               and r["brand"]["rank"] <= 3 and r["semantic"]["rank"] > 5]
for r, br, sr in clear_brand:
    print(f"  ✓ [{r['sub']:>12}] '{r['q'][:52]}' → {r['exp']}")
    print(f"           brand={br} sem={sr} | brand_top5: {r['brand']['top5']}")

# ── Semantic clearly better (semantic≤3, brand>5) ────────────────────────────
print("\n── SEMANTIC CLEAR WINS (semantic≤3, brand>5) ─────────────────────────")
clear_sem = [(r, r["brand"]["rank"], r["semantic"]["rank"])
             for r in rows if r.get("exp")
             and isinstance(r.get("brand",{}).get("rank"), int)
             and isinstance(r.get("semantic",{}).get("rank"), int)
             and r["semantic"]["rank"] <= 3 and r["brand"]["rank"] > 5]
for r, br, sr in clear_sem:
    print(f"  ✓ [{r['sub']:>12}] '{r['q'][:52]}' → {r['exp']}")
    print(f"           brand={br} sem={sr} | sem_top5: {r['semantic']['top5']}")

# ── Brand failures (brand>10 when expected given) ────────────────────────────
print("\n── BRAND FAILURES (brand rank > 10) ──────────────────────────────────")
brand_fails = [r for r in rows if r.get("exp")
               and isinstance(r.get("brand",{}).get("rank"), int)
               and r["brand"]["rank"] > 10]
print(f"  Total brand failures (rank>10): {len(brand_fails)}/{len(rows)}")
for r in brand_fails:
    br = r["brand"]["rank"]; sr = r["semantic"]["rank"]; ar = r["auto"]["rank"]
    print(f"  [{r['sub']:>12}] rank={br:>3} '{r['q'][:50]}' → {r['exp']}")
    print(f"             brand_top5: {r['brand']['top5'][:3]}")

# ── Palau-like dominance: countries appearing in top5 disproportionately ─────
print("\n── COUNTRY OVER-REPRESENTATION IN TOP-5 (brand mode) ─────────────────")
country_count = {}
for r in rows:
    for c in r.get("brand",{}).get("top5",[]):
        country_count[c] = country_count.get(c, 0) + 1
sorted_counts = sorted(country_count.items(), key=lambda x: -x[1])
print("  Top-20 most frequent in brand top-5 (across 110 travel queries):")
for country, cnt in sorted_counts[:20]:
    pct = 100 * cnt // len(rows)
    bar = "█" * (cnt // 2)
    print(f"  {country:<35} {cnt:>3}x ({pct:>2}%) {bar}")

print("\n── SAME for SEMANTIC mode ─────────────────────────────────────────────")
country_count_sem = {}
for r in rows:
    for c in r.get("semantic",{}).get("top5",[]):
        country_count_sem[c] = country_count_sem.get(c, 0) + 1
for country, cnt in sorted(country_count_sem.items(), key=lambda x: -x[1])[:15]:
    pct = 100 * cnt // len(rows)
    print(f"  {country:<35} {cnt:>3}x ({pct:>2}%)")

# ── Keyword pattern analysis: can we classify before search? ─────────────────
print("\n── KEYWORD PATTERN ANALYSIS ──────────────────────────────────────────")
BRAND_KEYWORDS = {
    "named brands": ["LEGO","IKEA","Ferrari","Bollywood","Hollywood","Nollywood",
                     "Michelin","Ritz","UNESCO"],
    "music/culture": ["reggae","kpop","K-pop","tango","flamenco","samba","fado",
                      "anime","manga","carnival"],
    "named inventions": ["overwater","bungee","steelpan","pysanky","Champagne"],
    "food/drink named": ["champagne","espresso","sake","kimchi","sushi","ramen",
                         "miso","taco","jerk","pierogi","baklava","hummus"],
    "named event": ["Oktoberfest","Diwali","Carnival","Hanami","lantern festival",
                    "Yi Peng","Glastonbury","Route 66"],
}
SEMANTIC_KEYWORDS = {
    "generic activity": ["hiking","trekking","diving","snorkeling","surfing","camping",
                         "kayaking","climbing","cycling"],
    "generic nature": ["beach","reef","coral","waterfall","glacier","volcano","fjord",
                       "forest","jungle","desert"],
    "generic geography": ["island","coast","mountain","valley","plateau","river"],
    "generic culture": ["temple","ruins","castle","palace","museum","market","bazaar"],
}

print("  Testing if 'brand keyword present' predicts brand winning …")
brand_kw_queries = []
no_brand_kw_queries = []
all_brand_kws = [kw.lower() for lst in BRAND_KEYWORDS.values() for kw in lst]
for r in rows:
    ql = r["q"].lower()
    has_brand_kw = any(kw in ql for kw in all_brand_kws)
    if has_brand_kw:
        brand_kw_queries.append(r)
    else:
        no_brand_kw_queries.append(r)

bh_with, n_with = hits_at(brand_kw_queries, "brand", 3)
bh_without, n_without = hits_at(no_brand_kw_queries, "brand", 3)
sh_with, _ = hits_at(brand_kw_queries, "semantic", 3)
sh_without, _ = hits_at(no_brand_kw_queries, "semantic", 3)

print(f"  Queries WITH brand keyword ({n_with}): brand_h@3={bh_with}/{n_with}={100*bh_with//n_with if n_with else 0}%  sem_h@3={sh_with}/{n_with}={100*sh_with//n_with if n_with else 0}%")
print(f"  Queries WITHOUT brand keyword ({n_without}): brand_h@3={bh_without}/{n_without}={100*bh_without//n_without if n_without else 0}%  sem_h@3={sh_without}/{n_without}={100*sh_without//n_without if n_without else 0}%")

# ── Simulated router: if brand keyword → brand, else semantic ────────────────
print("\n── SIMULATED TRAVEL ROUTER: brand_kw→brand, else→semantic ───────────")
router_hits_1 = 0; router_hits_3 = 0; router_hits_5 = 0; router_n = 0
baseline_hits_1 = 0; baseline_hits_3 = 0; baseline_hits_5 = 0
for r in rows:
    if not r.get("exp"): continue
    ql = r["q"].lower()
    has_brand_kw = any(kw in ql for kw in all_brand_kws)
    chosen_mode = "brand" if has_brand_kw else "semantic"
    rk = r.get(chosen_mode, {}).get("rank")
    base_rk = r.get("auto", {}).get("rank")  # baseline = auto
    if rk is None: continue
    router_n += 1
    if rk <= 1: router_hits_1 += 1
    if rk <= 3: router_hits_3 += 1
    if rk <= 5: router_hits_5 += 1
    if base_rk and base_rk <= 1: baseline_hits_1 += 1
    if base_rk and base_rk <= 3: baseline_hits_3 += 1
    if base_rk and base_rk <= 5: baseline_hits_5 += 1

print(f"  Travel router (n={router_n}):")
print(f"    Hit@1 : router={router_hits_1}/{router_n}={100*router_hits_1//router_n}%  baseline(auto)={baseline_hits_1}/{router_n}={100*baseline_hits_1//router_n}%  Δ={router_hits_1-baseline_hits_1:+}")
print(f"    Hit@3 : router={router_hits_3}/{router_n}={100*router_hits_3//router_n}%  baseline(auto)={baseline_hits_3}/{router_n}={100*baseline_hits_3//router_n}%  Δ={router_hits_3-baseline_hits_3:+}")
print(f"    Hit@5 : router={router_hits_5}/{router_n}={100*router_hits_5//router_n}%  baseline(auto)={baseline_hits_5}/{router_n}={100*baseline_hits_5//router_n}%  Δ={router_hits_5-baseline_hits_5:+}")

# ── Router wrong-choice breakdown ────────────────────────────────────────────
print("\n── ROUTER ERRORS: cases where router chose wrong mode ────────────────")
router_errors = []
for r in rows:
    if not r.get("exp"): continue
    ql = r["q"].lower()
    has_brand_kw = any(kw in ql for kw in all_brand_kws)
    chosen_mode = "brand" if has_brand_kw else "semantic"
    other_mode  = "semantic" if has_brand_kw else "brand"
    chosen_rk = r.get(chosen_mode, {}).get("rank")
    other_rk  = r.get(other_mode, {}).get("rank")
    if chosen_rk and other_rk and other_rk < chosen_rk and other_rk <= 3:
        router_errors.append((r, chosen_mode, chosen_rk, other_mode, other_rk))

print(f"  Router errors (chose wrong, other mode gave top-3): {len(router_errors)}")
for r, cm, cr, om, orr in router_errors:
    print(f"  [{r['sub']:>12}] chose={cm}(rank={cr}) better={om}(rank={orr}) | '{r['q'][:50]}'")
    print(f"             exp={r['exp']} | {cm}_top5: {r[cm]['top5'][:3]}")

# ── Confidence analysis for travel queries ────────────────────────────────────
print("\n── CONFIDENCE BADGE CALIBRATION (travel queries) ─────────────────────")
for mode in ["auto","brand","semantic"]:
    print(f"\n  Mode: {mode}")
    for cl in ["HIGH","MED","LOW"]:
        sub = [r for r in rows if r.get(mode,{}).get("conf") == cl and r.get("exp")]
        h1,n = hits_at(sub, mode, 1); h3,_ = hits_at(sub, mode, 3)
        print(f"    {cl:<4}: {n:>3} queries  hit@1={h1}/{n}({100*h1//n if n else 0}%)  hit@3={h3}/{n}({100*h3//n if n else 0}%)")

print("\n" + "=" * 76)
print("END OF TRAVEL ANALYSIS REPORT")
print("=" * 76)
