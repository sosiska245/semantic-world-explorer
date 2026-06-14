"""Multi-config quality benchmark: 125 queries, 6 configurations.

Configs tested:
  baseline    : multivec alpha=0.30, no length penalty (current state)
  penalty     : multivec alpha=0.30, + small-territory length penalty
  a0.10       : multivec alpha=0.10, no penalty
  a0.15       : multivec alpha=0.15, no penalty
  a0.20       : multivec alpha=0.20, no penalty
  a0.15+pen   : multivec alpha=0.15, + penalty  (likely best combo)

No API calls beyond the 125 query embeddings (same as original audit).
Run: .venv/bin/python scripts/run_audit_v2.py 2>&1 | tee audit_v2.txt
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from src.data_loader import ENTITIES_DF, EMBEDDINGS, N_ENTITIES, SECTION_EMBEDDINGS
from src.config import EMBEDDING_DIM, EMBEDDING_MODEL
from src.multivec import _MATS
from src.bm25 import bm25_scores as _bm25_raw

import voyageai
_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# ── precomputed arrays ────────────────────────────────────────────────────────
PROFILE_LENS = ENTITIES_DF["profile_text"].str.len().values.astype(np.float32)
_ISO3_TO_IDX = {iso3: i for i, iso3 in enumerate(ENTITIES_DF["iso3"].values)}
_ISO3_SET    = set(ENTITIES_DF["iso3"].values)

# ── small-territory penalty ───────────────────────────────────────────────────
# Countries with short profiles score moderately across unrelated queries
# ("hub" effect). Penalty proportional to profile length, exempt top-5 raw.
PENALTY_TARGET = 7000   # chars → penalty = 1.0 (no suppression)
TOP_EXEMPT     = 5      # protect top-N raw-score results from penalty

def apply_penalty(scores: np.ndarray) -> np.ndarray:
    pen = np.minimum(1.0, PROFILE_LENS / PENALTY_TARGET)
    top_raw = np.argsort(scores)[::-1][:TOP_EXEMPT]
    pen[top_raw] = 1.0
    return scores * pen

# ── multi-section similarity (alpha-parameterised) ────────────────────────────
def multi_sim_alpha(query_vec: np.ndarray, alpha: float) -> np.ndarray:
    main = EMBEDDINGS @ query_vec
    if alpha == 0.0:
        return main
    section_maxes = np.full(N_ENTITIES, -np.inf, dtype=np.float32)
    has_any = np.zeros(N_ENTITIES, dtype=bool)
    for key, (mat, has_data) in _MATS.items():
        if not has_data.any():
            continue
        s = mat @ query_vec
        improved = has_data & (s > section_maxes)
        section_maxes[improved] = s[improved]
        has_any[improved] = True
    result = main.copy()
    result[has_any] = (1 - alpha) * main[has_any] + alpha * section_maxes[has_any]
    return result

# ── configs ───────────────────────────────────────────────────────────────────
# Each config: (multivec_alpha, apply_penalty)
CONFIGS = [
    ("baseline",  0.30, False),
    ("penalty",   0.30, True),
    ("a0.10",     0.10, False),
    ("a0.15",     0.15, False),
    ("a0.20",     0.20, False),
    ("a0.15+pen", 0.15, True),
]

# ── queries (identical to run_audit.py) ──────────────────────────────────────
QUERIES = [
    ("Islam and Muslim culture",              "religion",    ["SAU","IRN","PAK","IDN","EGY","TUR","MYS","BGD","MAR","NGA"]),
    ("Buddhism and Buddhist monks",            "religion",    ["THA","MMR","LKA","BTN","KHM","LAO","MNG","JPN","CHN","VNM"]),
    ("Hinduism and Hindu temples",             "religion",    ["IND","NPL","IDN","MUS","FJI","SGP"]),
    ("Roman Catholic Christianity",            "religion",    ["ITA","ESP","PRT","BRA","PHL","POL","ARG","MEX","IRL","COL"]),
    ("Orthodox Christianity",                  "religion",    ["RUS","GRC","SRB","BGR","ROU","UKR","GEO","ARM","CYP","MKD"]),
    ("Jewish culture and Judaism",             "religion",    ["ISR","USA","HUN","AUT","DEU","FRA","GBR","ARG"]),
    ("Evangelical Protestant church",          "religion",    ["USA","BRA","KOR","NGA","ZAF","KEN","GHA","AUS","CAN"]),
    ("Secular atheist society",                "religion",    ["CZE","EST","FRA","DNK","SWE","NLD","NOR","FIN","DEU"]),
    ("liberal democracy and rule of law",      "politics",    ["NOR","FIN","SWE","DNK","NLD","NZL","CAN","AUS","CHE","DEU"]),
    ("authoritarian regime one-party state",   "politics",    ["CHN","PRK","CUB","VEN","BLR","RUS","TKM","ERI","SYR","AZE"]),
    ("constitutional monarchy",                "politics",    ["GBR","DNK","NOR","SWE","ESP","NLD","BEL","THA","JPN","MAR"]),
    ("theocracy religion and state",           "politics",    ["IRN","SAU","AFG","SOM","YEM","PAK","SDN","MRT","BHR"]),
    ("direct democracy and referendums",       "politics",    ["CHE","ISL","IRL","NZL","AUS","CAN","URY","DEU","SWE","FIN"]),
    ("military junta coup government",         "politics",    ["MMR","SDN","TCD","MLI","BFA","THA","EGY","PAK","TUR"]),
    ("elections and voting rights",            "politics",    ["SWE","FIN","NOR","NZL","CAN","AUS","DEU","GBR","FRA","USA"]),
    ("corruption and governance failure",      "politics",    ["SOM","SSD","SYR","VEN","NGA","CAF","HTI","AFG","IRQ"]),
    ("high GDP per capita wealthy economy",    "economy",     ["LUX","CHE","NOR","USA","SGP","QAT","AUS","DNK","SWE","NLD"]),
    ("manufacturing and industrial economy",   "economy",     ["CHN","DEU","JPN","KOR","USA","MEX","THA","POL","CZE"]),
    ("subsistence farming and poverty",        "economy",     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"]),
    ("financial services banking hub",         "economy",     ["CHE","LUX","SGP","GBR","USA","ARE","MUS"]),
    ("centrally planned communist economy",    "economy",     ["CHN","CUB","PRK","VNM","LAO","BLR","VEN","ERI","SYR","TKM"]),
    ("tourism dependent economy",              "economy",     ["MDV","BHS","ATG","VUT","FJI","WSM","TON","CPV"]),
    ("remittance diaspora dependent economy",  "economy",     ["TON","KGZ","TJK","SLV","NPL","GEO","JAM"]),
    ("venture capital startups technology hub","finance",     ["USA","GBR","ISR","SGP","DEU","FRA","IND","CAN","SWE"]),
    ("stock exchange financial market",        "finance",     ["USA","GBR","JPN","DEU","FRA","CHN","AUS","CHE","SGP"]),
    ("cryptocurrency blockchain adoption",     "finance",     ["EST","CHE","SGP","USA","GBR","NLD","DEU","CAN"]),
    ("fintech innovation mobile banking",      "finance",     ["GBR","SWE","SGP","USA","EST","KEN","NGA","IND","NLD","ISR"]),
    ("Silicon Valley software engineering",    "finance",     ["USA","ISR","IND","SGP","GBR","DEU","FIN","SWE","CAN","NLD"]),
    ("semiconductor chip manufacturing",       "finance",     ["KOR","JPN","USA","DEU","NLD","CHN","MYS","SGP","IRL"]),
    ("artificial intelligence research",       "technology",  ["USA","CHN","GBR","CAN","DEU","FRA","ISR","JPN","KOR","SGP"]),
    ("internet users digital society",         "technology",  ["NOR","ISL","DNK","FIN","SWE","NLD","CHE","GBR","KOR","LUX"]),
    ("social media mobile internet",           "technology",  ["KOR","USA","GBR","SGP","NLD","SWE","JPN","AUS","NZL","CAN"]),
    ("e-commerce online shopping",             "technology",  ["CHN","USA","GBR","DEU","KOR","JPN","AUS","NLD","FRA","SWE"]),
    ("digital government e-government",        "technology",  ["EST","DNK","NOR","FIN","SGP","KOR","AUS","NLD","CHE","SWE"]),
    ("university research excellence",         "education",   ["USA","GBR","DEU","FRA","JPN","CHN","CAN","AUS","CHE","SWE"]),
    ("high literacy rate education",           "education",   ["FIN","NOR","AUS","CAN","NZL","JPN","KOR","DEU","CHE","SWE"]),
    ("free tuition public university",         "education",   ["NOR","SWE","FIN","DEU","AUT","FRA","DNK","CZE","GRC","POL"]),
    ("PISA school rankings children",          "education",   ["SGP","FIN","JPN","KOR","CAN","EST","NLD","POL","SWE","CHE"]),
    ("vocational training apprenticeship",     "education",   ["DEU","AUT","CHE","DNK","NLD","FIN","NOR","SWE","GBR","JPN"]),
    ("universal healthcare system",            "health",      ["NOR","SWE","FIN","AUS","CAN","DEU","FRA","JPN","NZL","CHE"]),
    ("life expectancy longevity",              "health",      ["JPN","CHE","ESP","ITA","AUS","SGP","ISL","LUX","NOR","SWE"]),
    ("malaria disease tropical epidemic",      "health",      ["NGA","COD","TZA","MOZ","BFA","NER","MLI","CMR","UGA","GHA"]),
    ("medical tourism destination",            "health",      ["THA","IND","MYS","SGP","ISR","TUR","POL","CZE","HUN"]),
    ("tropical beach resort tourism",          "tourism",     ["MDV","THA","BHS","PHL","IDN","MEX","MYS","LKA","FJI","JAM"]),
    ("cultural heritage UNESCO tourism",       "tourism",     ["ITA","ESP","FRA","CHN","IND","MEX","GRC","EGY","JPN","PRT"]),
    ("adventure ecotourism wildlife",          "tourism",     ["CRI","BLZ","BOL","ECU","NZL","CAN","NOR","TZA","KEN"]),
    ("luxury cruise ship destination",         "tourism",     ["BHS","MDV","MEX","GRC","ITA","HRV","NOR","CYP","TTO","JAM"]),
    ("backpacker budget travel",               "tourism",     ["THA","VNM","IDN","IND","BOL","PER","MEX","LAO","KHM"]),
    ("safari wildlife game reserve",           "tourism",     ["TZA","KEN","ZAF","BWA","NAM","ZMB","ZWE","ETH","UGA","RWA"]),
    ("tropical island nation",                 "islands",     ["MDV","FJI","WSM","TON","VUT","NRU","KIR","COM","PLW"]),
    ("Caribbean island",                       "islands",     ["BHS","CUB","JAM","TTO","DOM","BRB","LCA","VCT","GRD","ATG"]),
    ("archipelago island chain",               "islands",     ["PHL","IDN","JPN","STP","CPV","MDV","FJI","VUT","KIR"]),
    ("coral reef marine biodiversity",         "islands",     ["AUS","PHL","IDN","MDV","FJI","PNG","MHL"]),
    ("volcanic island geology",                "islands",     ["ISL","IDN","ITA","PHL","ECU","CPV","VUT","TON","WSM","PNG"]),
    ("alpine skiing winter sports",            "mountains",   ["AUT","CHE","FRA","ITA","NOR","SWE","FIN","AND","LIE","SVN"]),
    ("high altitude Himalayan mountain",       "mountains",   ["NPL","CHN","IND","BTN","PAK","TJK","KGZ","AFG"]),
    ("fjords and glaciers landscape",          "mountains",   ["NOR","ISL","CHE","NZL","CAN","SVN","CHL"]),
    ("mountaineering trekking hiking",         "mountains",   ["NPL","CHE","NZL","NOR","PER","ECU","TZA","KEN","SVN"]),
    ("ski resort mountain tourism",            "mountains",   ["AUT","CHE","FRA","ITA","AND","LIE","SVK","BGR","KGZ","NOR"]),
    ("wheat grain cereal farming",             "agriculture", ["RUS","USA","CHN","IND","AUS","FRA","DEU","UKR","CAN","PAK"]),
    ("rice cultivation paddy field",           "agriculture", ["CHN","IND","IDN","BGD","VNM","THA","MMR","PHL","BRA","JPN"]),
    ("coffee production and export",           "agriculture", ["BRA","VNM","COL","ETH","IDN","UGA","MEX","IND","PER","HND"]),
    ("wine production viticulture",            "agriculture", ["FRA","ITA","ESP","PRT","ARG","AUS","CHL","ZAF","DEU","USA"]),
    ("cattle ranching beef livestock",         "agriculture", ["BRA","USA","AUS","ARG","MEX","ETH","CHN","RUS","COL","NZL"]),
    ("cacao cocoa chocolate production",       "agriculture", ["CIV","GHA","IDN","CMR","NGA","BRA","ECU","COD","UGA"]),
    ("spice tea plantation",                   "agriculture", ["IND","CHN","KEN","LKA","IDN","TZA","VNM","RWA","TUR","NPL"]),
    ("renewable solar wind energy",            "energy",      ["DNK","NOR","ISL","SWE","DEU","FIN","NZL","ESP","AUT","CHE"]),
    ("nuclear power electricity",              "energy",      ["FRA","USA","CHN","KOR","RUS","DEU","JPN","GBR","CAN","IND"]),
    ("hydroelectric dam power",                "energy",      ["NOR","BRA","CHN","CAN","ISL","PRY","VNM","COL","PER","ETH"]),
    ("electricity access energy poverty",      "energy",      ["ETH","NGA","TZA","UGA","MOZ","SSD","MWI","BFA","NER","SOM"]),
    ("clean energy transition",                "energy",      ["DNK","NOR","SWE","ISL","NZL","AUT","PRT","FIN","URY","GBR"]),
    ("solar photovoltaic panel",               "energy",      ["DEU","CHN","AUS","ESP","USA","IND","JPN","ITA","NLD","GBR"]),
    ("oil petroleum export country",           "oil_gas",     ["SAU","RUS","IRQ","IRN","ARE","KWT","NGA","VEN","AGO","LBY"]),
    ("natural gas pipeline export",            "oil_gas",     ["RUS","QAT","NOR","IRN","TKM","AZE","DZA","NGA","TTO","USA"]),
    ("OPEC oil cartel member",                 "oil_gas",     ["SAU","IRQ","IRN","ARE","KWT","NGA","VEN","AGO","LBY"]),
    ("Gulf state petrodollar wealth",          "oil_gas",     ["SAU","ARE","QAT","KWT","BHR","OMN","LBY","IRQ","IRN","AZE"]),
    ("tropical humid rainforest climate",      "climate",     ["BRA","COD","COG","IDN","MYS","PER","COL","ECU","CMR","PNG"]),
    ("arctic tundra subarctic",                "climate",     ["RUS","CAN","NOR","SWE","FIN","ISL","MNG","USA"]),
    ("monsoon rainy season flood",             "climate",     ["BGD","IND","MMR","THA","VNM","PHL","IDN","CHN","NPL","PAK"]),
    ("Mediterranean dry summer climate",       "climate",     ["GRC","ITA","ESP","PRT","TUN","LBY","MAR","ALB","CYP","HRV"]),
    ("climate change vulnerable country",      "climate",     ["BGD","MDV","KIR","TUV","MHL","NLD","VNM","NGA","ETH","SOM"]),
    ("Sahara desert arid landscape",           "deserts",     ["DZA","LBY","EGY","MRT","MAR","TUN","NER","TCD","MLI","SDN"]),
    ("desert nomad Bedouin camel",             "deserts",     ["SAU","OMN","MAR","DZA","EGY","IRQ","JOR","TUN","MRT"]),
    ("dryland water scarcity arid",            "deserts",     ["EGY","SAU","JOR","ISR","OMN","TUN","DZA","MAR","NAM","BWA"]),
    ("Amazon rainforest deforestation",        "forests",     ["BRA","PER","COL","ECU","BOL","VEN","GUY","SUR"]),
    ("boreal taiga forest",                    "forests",     ["RUS","CAN","SWE","FIN","NOR","USA","CHN","KAZ","MNG","EST"]),
    ("biodiversity endemic species",           "forests",     ["BRA","IDN","COL","AUS","MDG","PER","ECU","PNG","ZAF","VNM"]),
    ("military spending armed forces",         "military",    ["USA","CHN","RUS","SAU","IND","GBR","DEU","FRA","JPN","KOR"]),
    ("civil war armed conflict",               "military",    ["SYR","ETH","YEM","LBY","MLI","SSD","SOM","AFG","IRQ","SDN"]),
    ("peacekeeping UN troops",                 "military",    ["ETH","BGD","PAK","RWA","GHA","IND","NGA","FRA","URY","NLD"]),
    ("arms weapons export manufacturer",       "military",    ["USA","RUS","FRA","DEU","GBR","ISR","CHN","ITA","ESP","KOR"]),
    ("ancient civilization archaeology",       "history",     ["GRC","EGY","IRQ","IND","CHN","ITA","IRN","MEX","PER","TUR"]),
    ("World War II battlefield history",       "history",     ["DEU","FRA","POL","GBR","RUS","ITA","USA","NLD","BEL","GRC"]),
    ("Cold War Soviet communist bloc",         "history",     ["RUS","CHN","CUB","PRK","VNM","POL","HUN","CZE","BGR","ROU"]),
    ("colonial history empire",                "history",     ["GBR","FRA","ESP","PRT","NLD","BEL","DEU","ITA","RUS","JPN"]),
    ("Viking Norse history",                   "history",     ["NOR","SWE","DNK","ISL","GBR","IRL"]),
    ("indigenous native culture",              "history",     ["MEX","PER","BOL","ECU","GTM","CAN","AUS","NZL","BRA","COL"]),
    ("communist revolutionary socialist state","socialism",   ["CHN","CUB","PRK","VNM","LAO","VEN","BOL","ERI","BLR"]),
    ("Marxist left wing politics",             "socialism",   ["CHN","CUB","VEN","BOL","ECU","NIC","ZWE","MOZ","AGO","ETH"]),
    ("Arab Spring revolution protest",         "socialism",   ["TUN","EGY","LBY","SYR","YEM","BHR","MAR","JOR","SDN","IRQ"]),
    ("fashion design luxury brands",           "culture",     ["FRA","ITA","USA","GBR","ESP","DEU","JPN","BEL","CHE","SWE"]),
    ("classical music opera symphony",         "culture",     ["AUT","DEU","ITA","RUS","FRA","CZE","HUN","POL","GBR","USA"]),
    ("jazz blues rock music",                  "culture",     ["USA","GBR","JAM","CUB","BRA","FRA","IRL","AUS","CAN","NLD"]),
    ("film cinema movie industry",             "culture",     ["USA","IND","GBR","FRA","ITA","JPN","CHN","KOR","IRN","EGY"]),
    ("anime manga cartoon Japan",              "culture",     ["JPN","KOR","CHN","USA","FRA","GBR","CAN","AUS","DEU"]),
    ("gaming esports video games",             "culture",     ["KOR","USA","CHN","SWE","FIN","DEU","JPN","GBR","CAN","AUS"]),
    ("coffee culture cafe",                    "food",        ["ETH","ITA","BRA","COL","VNM","TUR","GRC","AUT","UGA","HND"]),
    ("wine culture viticulture",               "food",        ["FRA","ITA","ESP","PRT","ARG","AUS","CHL","ZAF","DEU","GRC"]),
    ("spicy street food cuisine",              "food",        ["THA","IND","MEX","VNM","IDN","ETH","KOR","TUR","MAR","LAO"]),
    ("beer brewing craft beer",                "food",        ["DEU","CZE","BEL","GBR","IRL","AUT","USA","AUS","NLD","POL"]),
    ("tea culture ceremony",                   "food",        ["CHN","JPN","IND","GBR","TUR","IRN","MAR","EGY","KEN","TZA"]),
    ("football soccer culture",                "sports",      ["BRA","ARG","ESP","DEU","FRA","ITA","GBR","NLD","PRT","BEL"]),
    ("cricket national sport",                 "sports",      ["IND","AUS","GBR","PAK","NZL","LKA","ZAF","BGD"]),
    ("cycling Tour de France",                 "sports",      ["FRA","BEL","ITA","ESP","NLD","GBR","AUS","COL","DNK","CHE"]),
    ("basketball NBA sport",                   "sports",      ["USA","LTU","SVN","FRA","AUS","CAN","ARG","SRB","NGA","TUR"]),
    ("Formula One motorsport racing",          "sports",      ["GBR","ITA","FRA","BRA","MCO","AUT","AUS","BHR","SGP"]),
    ("rugby union",                            "sports",      ["NZL","AUS","ZAF","FRA","IRL","ARG","FJI"]),
    ("major airline international hub airport","aviation",    ["ARE","SGP","GBR","USA","DEU","NLD","JPN","TUR","FRA"]),
    ("aviation airports",                      "aviation",    ["NLD","GBR","ARE","SGP","TUR","DEU","USA","JPN","FRA"]),
    ("container shipping maritime trade",      "aviation",    ["SGP","CHN","NLD","GBR","USA","DEU","ARE","KOR","JPN"]),
    ("landlocked country no sea access",       "aviation",    ["CHE","AUT","CZE","SVK","HUN","SRB","BOL","PRY","ETH","UGA"]),
    ("earthquake seismic activity",            "disasters",   ["JPN","TUR","IRN","IND","IDN","CHL","MEX","GRC","ITA","NZL"]),
    ("hurricane typhoon cyclone",              "disasters",   ["PHL","JPN","BGD","IND","CHN","USA","MEX","MDG","MOZ","CUB"]),
    ("volcanic eruption lava",                 "disasters",   ["IDN","ITA","JPN","ISL","PHL","ECU","COL","MEX","PNG","CPV"]),
    ("flood river delta disaster",             "disasters",   ["BGD","CHN","IND","VNM","NLD","MOZ","NGA","PHL","PAK","USA"]),
]

# ── helpers ───────────────────────────────────────────────────────────────────
_vec_cache = {}

def embed(text):
    key = text.lower().strip()
    if key in _vec_cache:
        return _vec_cache[key]
    res = _client.embed([text], model=EMBEDDING_MODEL, input_type="query",
                        output_dimension=EMBEDDING_DIM)
    vec = np.array(res.embeddings[0], dtype=np.float32)
    _vec_cache[key] = vec
    time.sleep(0.2)
    return vec


def rank_of(scores, iso3):
    if iso3 not in _ISO3_TO_IDX:
        return None
    idx = _ISO3_TO_IDX[iso3]
    return int((scores > scores[idx]).sum()) + 1


def hits_at_k(scores_arr, expected_iso3, k=20):
    in_ds = [iso3 for iso3 in expected_iso3 if iso3 in _ISO3_TO_IDX]
    if not in_ds:
        return 0, 0
    h = sum(1 for iso3 in in_ds if rank_of(scores_arr, iso3) <= k)
    return h, len(in_ds)


def small_hub_count(scores, expected_iso3, topn=20):
    """Count unexpected short-profile countries in top-N."""
    top_idx = np.argsort(scores)[::-1][:topn]
    count = 0
    for i in top_idx:
        iso3 = ENTITIES_DF.iloc[i]["iso3"]
        if PROFILE_LENS[i] < 5000 and iso3 not in expected_iso3:
            count += 1
    return count


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    n_q = len(QUERIES)
    print(f"Benchmark v2: {n_q} queries × {len(CONFIGS)} configs\n")

    # warn about ISO3 codes missing from dataset
    all_exp = set()
    for _, _, exp in QUERIES:
        all_exp.update(exp)
    missing = sorted(all_exp - _ISO3_SET)
    if missing:
        print(f"ISO3 not in dataset (skipped in counting): {missing}\n")

    # accumulate: config_name → {total_hits, total_exp, better_vs_sem, worse_vs_sem, hub_suppressions}
    # We treat "a0.30 no penalty" (baseline) semantic as fixed reference
    cfg_names   = [c[0] for c in CONFIGS]
    totals_hits = {n: 0 for n in cfg_names}
    totals_exp  = {n: 0 for n in cfg_names}
    better      = {n: 0 for n in cfg_names}   # queries where this config > baseline
    worse       = {n: 0 for n in cfg_names}   # queries where this config < baseline
    hub_total   = {n: 0 for n in cfg_names}   # total hub contamination across queries
    per_cat     = {n: {} for n in cfg_names}

    # per-query detail rows (compact)
    detail_rows = []

    for qi, (query_text, category, expected_iso3) in enumerate(QUERIES, 1):
        print(f"  [{qi:03d}/{n_q}] {query_text[:65]}", flush=True)
        vec = embed(query_text)

        # compute scores for all configs
        config_scores = {}
        for cfg_name, alpha, use_penalty in CONFIGS:
            s = multi_sim_alpha(vec, alpha)
            if use_penalty:
                s = apply_penalty(s)
            config_scores[cfg_name] = s

        # semantic = baseline config (alpha=0.30, no penalty)
        sem_scores = config_scores["baseline"]

        row = {"q": query_text, "cat": category}
        baseline_hits, total = hits_at_k(sem_scores, expected_iso3)

        for cfg_name, alpha, use_penalty in CONFIGS:
            s = config_scores[cfg_name]
            h, t = hits_at_k(s, expected_iso3)
            totals_hits[cfg_name] += h
            totals_exp[cfg_name]  += t
            hub_total[cfg_name]   += small_hub_count(s, expected_iso3)

            # vs baseline
            if h > baseline_hits:
                better[cfg_name] += 1
            elif h < baseline_hits:
                worse[cfg_name] += 1

            # per-category
            cat = category
            if cat not in per_cat[cfg_name]:
                per_cat[cfg_name][cat] = [0, 0]
            per_cat[cfg_name][cat][0] += h
            per_cat[cfg_name][cat][1] += t

            row[cfg_name] = h

        row["total"] = total
        detail_rows.append(row)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'#'*90}")
    print(f"  SUMMARY — {n_q} queries, hits@20")
    print(f"{'#'*90}\n")

    # reference: baseline hits
    base_total = totals_hits["baseline"]
    base_denom = totals_exp["baseline"]
    base_pct   = 100 * base_total / base_denom

    print(f"  {'Config':<14} {'Hits':>6} {'Denom':>6} {'Rate':>7}  {'vs baseline':>12}  {'better':>7}  {'worse':>7}  {'hubs@20':>8}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*7}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*8}")
    for cfg_name, alpha, use_penalty in CONFIGS:
        h   = totals_hits[cfg_name]
        d   = totals_exp[cfg_name]
        pct = 100 * h / d
        delta = pct - base_pct
        sign  = "+" if delta >= 0 else ""
        b_cnt = better[cfg_name]
        w_cnt = worse[cfg_name]
        hub_c = hub_total[cfg_name]
        marker = " ◄ BEST" if h == max(totals_hits.values()) else ""
        print(f"  {cfg_name:<14} {h:>6} {d:>6} {pct:>6.1f}%  {sign}{delta:>+.1f} pp       {b_cnt:>7}  {w_cnt:>7}  {hub_c:>8}{marker}")

    # ── per-category breakdown ────────────────────────────────────────────────
    all_cats = sorted({r["cat"] for r in detail_rows})
    print(f"\n\n  Per-category hits@20 (% = hits/total_expected in that category)")
    hdr = f"  {'Category':<16}" + "".join(f" {n:>9}" for n in cfg_names)
    print(hdr)
    print("  " + "-"*16 + "".join(" " + "-"*9 for _ in cfg_names))
    for cat in all_cats:
        row_str = f"  {cat:<16}"
        for cfg_name in cfg_names:
            h, t = per_cat[cfg_name].get(cat, [0, 1])
            pct = 100 * h / max(t, 1)
            row_str += f" {pct:>8.0f}%"
        print(row_str)

    # ── per-query delta table (queries where configs diverge most) ────────────
    print(f"\n\n  Per-query results (queries where any config differs from baseline)")
    print(f"  {'#':<4} {'Cat':<12} {'Query':<45}" + "".join(f" {n:>9}" for n in cfg_names))
    print("  " + "-"*4 + " " + "-"*12 + " " + "-"*45 + "".join(" " + "-"*9 for _ in cfg_names))
    for i, r in enumerate(detail_rows, 1):
        base_h = r["baseline"]
        if any(r[n] != base_h for n in cfg_names):
            row_str = f"  {i:<4} {r['cat']:<12} {r['q'][:44]:<45}"
            for n in cfg_names:
                h = r[n]
                t = r["total"]
                diff = h - base_h
                tag = f"{h}/{t}"
                if diff > 0:
                    tag = f"+{tag}"
                elif diff < 0:
                    tag = f"={tag}"
                else:
                    tag = f" {tag}"
                row_str += f" {tag:>9}"
            print(row_str)

    # ── winner summary ────────────────────────────────────────────────────────
    print(f"\n\n  WINNER BY OVERALL HITS@20:")
    ranked = sorted(cfg_names, key=lambda n: totals_hits[n], reverse=True)
    for rank, n in enumerate(ranked, 1):
        h   = totals_hits[n]
        pct = 100 * h / totals_exp[n]
        delta = pct - base_pct
        sign  = "+" if delta >= 0 else ""
        print(f"    #{rank}  {n:<14}  {h} hits  {pct:.1f}%  ({sign}{delta:.1f} pp vs baseline)")

    print(f"\n  NOTE: 'baseline' = multivec alpha=0.30, no length penalty (current app behaviour)")
    print(f"  NOTE: all configs use multivec (section boost), NOT pure semantic.")
    print(f"  Semantic pure (EMBEDDINGS @ vec only) not shown here — see audit_results.txt for reference (42.7%).")
    print(f"\nBenchmark complete.")


if __name__ == "__main__":
    main()
