"""Quality audit: 100 queries across 34 categories.
Compares semantic / multivec / hybrid modes, detects root causes, reports.
Run from project root: .venv/bin/python scripts/run_audit.py 2>&1 | tee audit_results.txt
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── data ──────────────────────────────────────────────────────────────────────
from src.data_loader import ENTITIES_DF, EMBEDDINGS, N_ENTITIES, SECTION_EMBEDDINGS
from src.config import EMBEDDING_DIM, EMBEDDING_MODEL
from src.multivec import multi_sim, _MATS
from src.bm25 import bm25_scores, blend as bm25_blend

import voyageai
_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

PROFILE_LENS = ENTITIES_DF["profile_text"].str.len().values

_ISO3_SET = set(ENTITIES_DF["iso3"].values)
_ISO3_TO_IDX = {iso3: i for i, iso3 in enumerate(ENTITIES_DF["iso3"].values)}
_ISO3_TO_NAME = dict(zip(ENTITIES_DF["iso3"], ENTITIES_DF["name"]))

# section text lengths per section key
_SEC_TEXT_LENS = {}
for key in ["economy","culture","geography","politics","history"]:
    col = f"text_{key}"
    if col in ENTITIES_DF.columns:
        _SEC_TEXT_LENS[key] = ENTITIES_DF[col].fillna("").str.len().values
    else:
        _SEC_TEXT_LENS[key] = np.zeros(N_ENTITIES)

# ── query catalogue ───────────────────────────────────────────────────────────
# (query_text, category, expected_iso3_list)
QUERIES = [
    # -- religion (8) --
    ("Islam and Muslim culture",              "religion",    ["SAU","IRN","PAK","IDN","EGY","TUR","MYS","BGD","MAR","NGA"]),
    ("Buddhism and Buddhist monks",            "religion",    ["THA","MMR","LKA","BTN","KHM","LAO","MNG","JPN","CHN","VNM"]),
    ("Hinduism and Hindu temples",             "religion",    ["IND","NPL","IDN","MUS","FJI","SGP"]),
    ("Roman Catholic Christianity",            "religion",    ["ITA","ESP","PRT","BRA","PHL","POL","ARG","MEX","IRL","COL"]),
    ("Orthodox Christianity",                  "religion",    ["RUS","GRC","SRB","BGR","ROU","UKR","GEO","ARM","CYP","MKD"]),
    ("Jewish culture and Judaism",             "religion",    ["ISR","USA","HUN","AUT","DEU","FRA","GBR","ARG"]),
    ("Evangelical Protestant church",          "religion",    ["USA","BRA","KOR","NGA","ZAF","KEN","GHA","AUS","CAN"]),
    ("Secular atheist society",                "religion",    ["CZE","EST","FRA","DNK","SWE","NLD","NOR","FIN","DEU"]),

    # -- politics / democracy (8) --
    ("liberal democracy and rule of law",      "politics",    ["NOR","FIN","SWE","DNK","NLD","NZL","CAN","AUS","CHE","DEU"]),
    ("authoritarian regime one-party state",   "politics",    ["CHN","PRK","CUB","VEN","BLR","RUS","TKM","ERI","SYR","AZE"]),
    ("constitutional monarchy",                "politics",    ["GBR","DNK","NOR","SWE","ESP","NLD","BEL","THA","JPN","MAR"]),
    ("theocracy religion and state",           "politics",    ["IRN","SAU","AFG","SOM","YEM","PAK","SDN","MRT","BHR"]),
    ("direct democracy and referendums",       "politics",    ["CHE","ISL","IRL","NZL","AUS","CAN","URY","DEU","SWE","FIN"]),
    ("military junta coup government",         "politics",    ["MMR","SDN","TCD","MLI","BFA","THA","EGY","PAK","TUR"]),
    ("elections and voting rights",            "politics",    ["SWE","FIN","NOR","NZL","CAN","AUS","DEU","GBR","FRA","USA"]),
    ("corruption and governance failure",      "politics",    ["SOM","SSD","SYR","VEN","NGA","CAF","HTI","AFG","IRQ"]),

    # -- economy (7) --
    ("high GDP per capita wealthy economy",    "economy",     ["LUX","CHE","NOR","USA","SGP","QAT","AUS","DNK","SWE","NLD"]),
    ("manufacturing and industrial economy",   "economy",     ["CHN","DEU","JPN","KOR","USA","MEX","THA","POL","CZE"]),
    ("subsistence farming and poverty",        "economy",     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"]),
    ("financial services banking hub",         "economy",     ["CHE","LUX","SGP","GBR","USA","ARE","MUS"]),
    ("centrally planned communist economy",    "economy",     ["CHN","CUB","PRK","VNM","LAO","BLR","VEN","ERI","SYR","TKM"]),
    ("tourism dependent economy",              "economy",     ["MDV","BHS","ATG","VUT","FJI","WSM","TON","CPV"]),
    ("remittance diaspora dependent economy",  "economy",     ["TON","KGZ","TJK","SLV","NPL","GEO","JAM"]),

    # -- finance / startups (6) --
    ("venture capital startups technology hub","finance",     ["USA","GBR","ISR","SGP","DEU","FRA","IND","CAN","SWE"]),
    ("stock exchange financial market",        "finance",     ["USA","GBR","JPN","DEU","FRA","CHN","AUS","CHE","SGP"]),
    ("cryptocurrency blockchain adoption",     "finance",     ["EST","CHE","SGP","USA","GBR","NLD","DEU","CAN"]),
    ("fintech innovation mobile banking",      "finance",     ["GBR","SWE","SGP","USA","EST","KEN","NGA","IND","NLD","ISR"]),
    ("Silicon Valley software engineering",    "finance",     ["USA","ISR","IND","SGP","GBR","DEU","FIN","SWE","CAN","NLD"]),
    ("semiconductor chip manufacturing",       "finance",     ["KOR","JPN","USA","DEU","NLD","CHN","MYS","SGP","IRL"]),

    # -- technology / AI / internet (5) --
    ("artificial intelligence research",       "technology",  ["USA","CHN","GBR","CAN","DEU","FRA","ISR","JPN","KOR","SGP"]),
    ("internet users digital society",         "technology",  ["NOR","ISL","DNK","FIN","SWE","NLD","CHE","GBR","KOR","LUX"]),
    ("social media mobile internet",           "technology",  ["KOR","USA","GBR","SGP","NLD","SWE","JPN","AUS","NZL","CAN"]),
    ("e-commerce online shopping",             "technology",  ["CHN","USA","GBR","DEU","KOR","JPN","AUS","NLD","FRA","SWE"]),
    ("digital government e-government",        "technology",  ["EST","DNK","NOR","FIN","SGP","KOR","AUS","NLD","CHE","SWE"]),

    # -- education (5) --
    ("university research excellence",         "education",   ["USA","GBR","DEU","FRA","JPN","CHN","CAN","AUS","CHE","SWE"]),
    ("high literacy rate education",           "education",   ["FIN","NOR","AUS","CAN","NZL","JPN","KOR","DEU","CHE","SWE"]),
    ("free tuition public university",         "education",   ["NOR","SWE","FIN","DEU","AUT","FRA","DNK","CZE","GRC","POL"]),
    ("PISA school rankings children",          "education",   ["SGP","FIN","JPN","KOR","CAN","EST","NLD","POL","SWE","CHE"]),
    ("vocational training apprenticeship",     "education",   ["DEU","AUT","CHE","DNK","NLD","FIN","NOR","SWE","GBR","JPN"]),

    # -- health (4) --
    ("universal healthcare system",            "health",      ["NOR","SWE","FIN","AUS","CAN","DEU","FRA","JPN","NZL","CHE"]),
    ("life expectancy longevity",              "health",      ["JPN","CHE","ESP","ITA","AUS","SGP","ISL","LUX","NOR","SWE"]),
    ("malaria disease tropical epidemic",      "health",      ["NGA","COD","TZA","MOZ","BFA","NER","MLI","CMR","UGA","GHA"]),
    ("medical tourism destination",            "health",      ["THA","IND","MYS","SGP","ISR","TUR","POL","CZE","HUN"]),

    # -- tourism (6) --
    ("tropical beach resort tourism",          "tourism",     ["MDV","THA","BHS","PHL","IDN","MEX","MYS","LKA","FJI","JAM"]),
    ("cultural heritage UNESCO tourism",       "tourism",     ["ITA","ESP","FRA","CHN","IND","MEX","GRC","EGY","JPN","PRT"]),
    ("adventure ecotourism wildlife",          "tourism",     ["CRI","BLZ","BOL","ECU","NZL","CAN","NOR","TZA","KEN"]),
    ("luxury cruise ship destination",         "tourism",     ["BHS","MDV","MEX","GRC","ITA","HRV","NOR","CYP","TTO","JAM"]),
    ("backpacker budget travel",               "tourism",     ["THA","VNM","IDN","IND","BOL","PER","MEX","LAO","KHM"]),
    ("safari wildlife game reserve",           "tourism",     ["TZA","KEN","ZAF","BWA","NAM","ZMB","ZWE","ETH","UGA","RWA"]),

    # -- islands / beaches (5) --
    ("tropical island nation",                 "islands",     ["MDV","FJI","WSM","TON","VUT","NRU","KIR","COM","PLW"]),
    ("Caribbean island",                       "islands",     ["BHS","CUB","JAM","TTO","DOM","BRB","LCA","VCT","GRD","ATG"]),
    ("archipelago island chain",               "islands",     ["PHL","IDN","JPN","STP","CPV","MDV","FJI","VUT","KIR"]),
    ("coral reef marine biodiversity",         "islands",     ["AUS","PHL","IDN","MDV","FJI","PNG","MHL"]),
    ("volcanic island geology",                "islands",     ["ISL","IDN","ITA","PHL","ECU","CPV","VUT","TON","WSM","PNG"]),

    # -- mountains / skiing (5) --
    ("alpine skiing winter sports",            "mountains",   ["AUT","CHE","FRA","ITA","NOR","SWE","FIN","AND","LIE","SVN"]),
    ("high altitude Himalayan mountain",       "mountains",   ["NPL","CHN","IND","BTN","PAK","TJK","KGZ","AFG"]),
    ("fjords and glaciers landscape",          "mountains",   ["NOR","ISL","CHE","NZL","CAN","SVN","CHL"]),
    ("mountaineering trekking hiking",         "mountains",   ["NPL","CHE","NZL","NOR","PER","ECU","TZA","KEN","SVN"]),
    ("ski resort mountain tourism",            "mountains",   ["AUT","CHE","FRA","ITA","AND","LIE","SVK","BGR","KGZ","NOR"]),

    # -- agriculture (7) --
    ("wheat grain cereal farming",             "agriculture", ["RUS","USA","CHN","IND","AUS","FRA","DEU","UKR","CAN","PAK"]),
    ("rice cultivation paddy field",           "agriculture", ["CHN","IND","IDN","BGD","VNM","THA","MMR","PHL","BRA","JPN"]),
    ("coffee production and export",           "agriculture", ["BRA","VNM","COL","ETH","IDN","UGA","MEX","IND","PER","HND"]),
    ("wine production viticulture",            "agriculture", ["FRA","ITA","ESP","PRT","ARG","AUS","CHL","ZAF","DEU","USA"]),
    ("cattle ranching beef livestock",         "agriculture", ["BRA","USA","AUS","ARG","MEX","ETH","CHN","RUS","COL","NZL"]),
    ("cacao cocoa chocolate production",       "agriculture", ["CIV","GHA","IDN","CMR","NGA","BRA","ECU","COD","UGA"]),
    ("spice tea plantation",                   "agriculture", ["IND","CHN","KEN","LKA","IDN","TZA","VNM","RWA","TUR","NPL"]),

    # -- energy (6) --
    ("renewable solar wind energy",            "energy",      ["DNK","NOR","ISL","SWE","DEU","FIN","NZL","ESP","AUT","CHE"]),
    ("nuclear power electricity",              "energy",      ["FRA","USA","CHN","KOR","RUS","DEU","JPN","GBR","CAN","IND"]),
    ("hydroelectric dam power",                "energy",      ["NOR","BRA","CHN","CAN","ISL","PRY","VNM","COL","PER","ETH"]),
    ("electricity access energy poverty",      "energy",      ["ETH","NGA","TZA","UGA","MOZ","SSD","MWI","BFA","NER","SOM"]),
    ("clean energy transition",                "energy",      ["DNK","NOR","SWE","ISL","NZL","AUT","PRT","FIN","URY","GBR"]),
    ("solar photovoltaic panel",               "energy",      ["DEU","CHN","AUS","ESP","USA","IND","JPN","ITA","NLD","GBR"]),

    # -- oil / gas (4) --
    ("oil petroleum export country",           "oil_gas",     ["SAU","RUS","IRQ","IRN","ARE","KWT","NGA","VEN","AGO","LBY"]),
    ("natural gas pipeline export",            "oil_gas",     ["RUS","QAT","NOR","IRN","TKM","AZE","DZA","NGA","TTO","USA"]),
    ("OPEC oil cartel member",                 "oil_gas",     ["SAU","IRQ","IRN","ARE","KWT","NGA","VEN","AGO","LBY"]),
    ("Gulf state petrodollar wealth",          "oil_gas",     ["SAU","ARE","QAT","KWT","BHR","OMN","LBY","IRQ","IRN","AZE"]),

    # -- climate (5) --
    ("tropical humid rainforest climate",      "climate",     ["BRA","COD","COG","IDN","MYS","PER","COL","ECU","CMR","PNG"]),
    ("arctic tundra subarctic",                "climate",     ["RUS","CAN","NOR","SWE","FIN","ISL","MNG","USA"]),
    ("monsoon rainy season flood",             "climate",     ["BGD","IND","MMR","THA","VNM","PHL","IDN","CHN","NPL","PAK"]),
    ("Mediterranean dry summer climate",       "climate",     ["GRC","ITA","ESP","PRT","TUN","LBY","MAR","ALB","CYP","HRV"]),
    ("climate change vulnerable country",      "climate",     ["BGD","MDV","KIR","TUV","MHL","NLD","VNM","NGA","ETH","SOM"]),

    # -- deserts (3) --
    ("Sahara desert arid landscape",           "deserts",     ["DZA","LBY","EGY","MRT","MAR","TUN","NER","TCD","MLI","SDN"]),
    ("desert nomad Bedouin camel",             "deserts",     ["SAU","OMN","MAR","DZA","EGY","IRQ","JOR","TUN","MRT"]),
    ("dryland water scarcity arid",            "deserts",     ["EGY","SAU","JOR","ISR","OMN","TUN","DZA","MAR","NAM","BWA"]),

    # -- forests (3) --
    ("Amazon rainforest deforestation",        "forests",     ["BRA","PER","COL","ECU","BOL","VEN","GUY","SUR"]),
    ("boreal taiga forest",                    "forests",     ["RUS","CAN","SWE","FIN","NOR","USA","CHN","KAZ","MNG","EST"]),
    ("biodiversity endemic species",           "forests",     ["BRA","IDN","COL","AUS","MDG","PER","ECU","PNG","ZAF","VNM"]),

    # -- military / conflict (4) --
    ("military spending armed forces",         "military",    ["USA","CHN","RUS","SAU","IND","GBR","DEU","FRA","JPN","KOR"]),
    ("civil war armed conflict",               "military",    ["SYR","ETH","YEM","LBY","MLI","SSD","SOM","AFG","IRQ","SDN"]),
    ("peacekeeping UN troops",                 "military",    ["ETH","BGD","PAK","RWA","GHA","IND","NGA","FRA","URY","NLD"]),
    ("arms weapons export manufacturer",       "military",    ["USA","RUS","FRA","DEU","GBR","ISR","CHN","ITA","ESP","KOR"]),

    # -- history (6) --
    ("ancient civilization archaeology",       "history",     ["GRC","EGY","IRQ","IND","CHN","ITA","IRN","MEX","PER","TUR"]),
    ("World War II battlefield history",       "history",     ["DEU","FRA","POL","GBR","RUS","ITA","USA","NLD","BEL","GRC"]),
    ("Cold War Soviet communist bloc",         "history",     ["RUS","CHN","CUB","PRK","VNM","POL","HUN","CZE","BGR","ROU"]),
    ("colonial history empire",                "history",     ["GBR","FRA","ESP","PRT","NLD","BEL","DEU","ITA","RUS","JPN"]),
    ("Viking Norse history",                   "history",     ["NOR","SWE","DNK","ISL","GBR","IRL"]),
    ("indigenous native culture",              "history",     ["MEX","PER","BOL","ECU","GTM","CAN","AUS","NZL","BRA","COL"]),

    # -- socialism / revolution (3) --
    ("communist revolutionary socialist state","socialism",   ["CHN","CUB","PRK","VNM","LAO","VEN","BOL","ERI","BLR"]),
    ("Marxist left wing politics",             "socialism",   ["CHN","CUB","VEN","BOL","ECU","NIC","ZWE","MOZ","AGO","ETH"]),
    ("Arab Spring revolution protest",         "socialism",   ["TUN","EGY","LBY","SYR","YEM","BHR","MAR","JOR","SDN","IRQ"]),

    # -- culture / arts (6) --
    ("fashion design luxury brands",           "culture",     ["FRA","ITA","USA","GBR","ESP","DEU","JPN","BEL","CHE","SWE"]),
    ("classical music opera symphony",         "culture",     ["AUT","DEU","ITA","RUS","FRA","CZE","HUN","POL","GBR","USA"]),
    ("jazz blues rock music",                  "culture",     ["USA","GBR","JAM","CUB","BRA","FRA","IRL","AUS","CAN","NLD"]),
    ("film cinema movie industry",             "culture",     ["USA","IND","GBR","FRA","ITA","JPN","CHN","KOR","IRN","EGY"]),
    ("anime manga cartoon Japan",              "culture",     ["JPN","KOR","CHN","USA","FRA","GBR","CAN","AUS","DEU"]),
    ("gaming esports video games",             "culture",     ["KOR","USA","CHN","SWE","FIN","DEU","JPN","GBR","CAN","AUS"]),

    # -- food / drink (5) --
    ("coffee culture cafe",                    "food",        ["ETH","ITA","BRA","COL","VNM","TUR","GRC","AUT","UGA","HND"]),
    ("wine culture viticulture",               "food",        ["FRA","ITA","ESP","PRT","ARG","AUS","CHL","ZAF","DEU","GRC"]),
    ("spicy street food cuisine",              "food",        ["THA","IND","MEX","VNM","IDN","ETH","KOR","TUR","MAR","LAO"]),
    ("beer brewing craft beer",                "food",        ["DEU","CZE","BEL","GBR","IRL","AUT","USA","AUS","NLD","POL"]),
    ("tea culture ceremony",                   "food",        ["CHN","JPN","IND","GBR","TUR","IRN","MAR","EGY","KEN","TZA"]),

    # -- sports (6) --
    ("football soccer culture",                "sports",      ["BRA","ARG","ESP","DEU","FRA","ITA","GBR","NLD","PRT","BEL"]),
    ("cricket national sport",                 "sports",      ["IND","AUS","GBR","PAK","NZL","LKA","ZAF","BGD"]),
    ("cycling Tour de France",                 "sports",      ["FRA","BEL","ITA","ESP","NLD","GBR","AUS","COL","DNK","CHE"]),
    ("basketball NBA sport",                   "sports",      ["USA","LTU","SVN","FRA","AUS","CAN","ARG","SRB","NGA","TUR"]),
    ("Formula One motorsport racing",          "sports",      ["GBR","ITA","FRA","BRA","MCO","AUT","AUS","BHR","SGP"]),
    ("rugby union",                            "sports",      ["NZL","AUS","ZAF","FRA","IRL","ARG","FJI"]),

    # -- aviation / transport (4) --
    ("major airline international hub airport","aviation",    ["ARE","SGP","GBR","USA","DEU","NLD","JPN","TUR","FRA"]),
    ("aviation airports",                      "aviation",    ["NLD","GBR","ARE","SGP","TUR","DEU","USA","JPN","FRA"]),
    ("container shipping maritime trade",      "aviation",    ["SGP","CHN","NLD","GBR","USA","DEU","ARE","KOR","JPN"]),
    ("landlocked country no sea access",       "aviation",    ["CHE","AUT","CZE","SVK","HUN","SRB","BOL","PRY","ETH","UGA"]),

    # -- natural disasters (4) --
    ("earthquake seismic activity",            "disasters",   ["JPN","TUR","IRN","IND","IDN","CHL","MEX","GRC","ITA","NZL"]),
    ("hurricane typhoon cyclone",              "disasters",   ["PHL","JPN","BGD","IND","CHN","USA","MEX","MDG","MOZ","CUB"]),
    ("volcanic eruption lava",                 "disasters",   ["IDN","ITA","JPN","ISL","PHL","ECU","COL","MEX","PNG","CPV"]),
    ("flood river delta disaster",             "disasters",   ["BGD","CHN","IND","VNM","NLD","MOZ","NGA","PHL","PAK","USA"]),
]

# category → most relevant section key for SECTION_TRUNCATION check
CAT_TO_SECTION = {
    "religion":    "culture",
    "politics":    "politics",
    "economy":     "economy",
    "finance":     "economy",
    "technology":  "economy",
    "education":   "culture",
    "health":      "culture",
    "tourism":     "culture",
    "islands":     "geography",
    "mountains":   "geography",
    "agriculture": "economy",
    "energy":      "economy",
    "oil_gas":     "economy",
    "climate":     "geography",
    "deserts":     "geography",
    "forests":     "geography",
    "military":    "politics",
    "history":     "history",
    "socialism":   "politics",
    "culture":     "culture",
    "food":        "culture",
    "sports":      "culture",
    "aviation":    "economy",
    "disasters":   "geography",
}

# ── embed + score ─────────────────────────────────────────────────────────────
_vec_cache = {}

def embed(text):
    key = text.lower().strip()
    if key in _vec_cache:
        return _vec_cache[key]
    res = _client.embed([text], model=EMBEDDING_MODEL, input_type="query", output_dimension=EMBEDDING_DIM)
    vec = np.array(res.embeddings[0], dtype=np.float32)
    _vec_cache[key] = vec
    time.sleep(0.2)
    return vec


def compute_scores(vec, query_text):
    semantic  = EMBEDDINGS @ vec
    multivec  = multi_sim(query_vec=vec, alpha=0.3)
    hybrid    = bm25_blend(multivec, query_text, alpha=0.25)
    return {"semantic": semantic, "multivec": multivec, "hybrid": hybrid}


def top_k(scores, k=20):
    idx = np.argsort(scores)[::-1][:k]
    return [(ENTITIES_DF.iloc[i]["iso3"], ENTITIES_DF.iloc[i]["name"], float(scores[i])) for i in idx]


def rank_of(scores, iso3):
    if iso3 not in _ISO3_TO_IDX:
        return None
    idx = _ISO3_TO_IDX[iso3]
    return int((scores > scores[idx]).sum()) + 1


# ── root cause detection ──────────────────────────────────────────────────────
SMALL_TERRITORY_THRESHOLD = 5000
SECTION_CAP = 3000
BM25_NOISE_RISE = 30

def flag_causes(top20_sem, top20_mv, top20_hyb, scores, expected_iso3, category):
    causes = set()
    section_key = CAT_TO_SECTION.get(category, "culture")

    # SMALL_TERRITORY_HUB: any of top20 in any mode has short profile
    for lst in [top20_sem, top20_mv, top20_hyb]:
        for iso3, name, score in lst[:20]:
            if iso3 not in _ISO3_TO_IDX:
                continue
            idx = _ISO3_TO_IDX[iso3]
            if PROFILE_LENS[idx] < SMALL_TERRITORY_THRESHOLD and iso3 not in expected_iso3:
                causes.add("SMALL_TERRITORY_HUB")
                break

    # SECTION_TRUNCATION: any expected country has section at cap
    for iso3 in expected_iso3:
        if iso3 not in _ISO3_TO_IDX:
            continue
        idx = _ISO3_TO_IDX[iso3]
        if _SEC_TEXT_LENS.get(section_key, np.zeros(N_ENTITIES))[idx] >= SECTION_CAP:
            causes.add(f"SECTION_TRUNCATION({section_key})")

    # MISSING_WIKI_SECTION: expected country has empty section
    for iso3 in expected_iso3:
        if iso3 not in _ISO3_TO_IDX:
            continue
        idx = _ISO3_TO_IDX[iso3]
        if _SEC_TEXT_LENS.get(section_key, np.zeros(N_ENTITIES))[idx] == 0:
            causes.add(f"MISSING_WIKI_SECTION({section_key}:{iso3})")

    # BM25_NOISE: country rises >=30 ranks hybrid vs multivec AND not in expected
    sem_ranks  = {iso3: rank_of(scores["semantic"], iso3)  for iso3, *_ in top20_sem[:20]}
    mv_ranks   = {iso3: rank_of(scores["multivec"], iso3)  for iso3, *_ in top20_mv[:20]}
    hyb_ranks  = {iso3: rank_of(scores["hybrid"],   iso3)  for iso3, *_ in top20_hyb[:20]}
    # check top20 hybrid for big rises vs multivec
    all_top_hyb_iso3 = [iso3 for iso3,*_ in top20_hyb[:20]]
    for iso3 in all_top_hyb_iso3:
        if iso3 in expected_iso3 or iso3 not in _ISO3_TO_IDX:
            continue
        mv_r = rank_of(scores["multivec"], iso3)
        h_r  = rank_of(scores["hybrid"],   iso3)
        if mv_r is not None and h_r is not None and mv_r - h_r >= BM25_NOISE_RISE:
            causes.add("BM25_NOISE")
            break

    # SEMANTIC_MISS: expected country >50 in ALL modes
    in_dataset = [iso3 for iso3 in expected_iso3 if iso3 in _ISO3_TO_IDX]
    if in_dataset:
        all_miss = all(
            rank_of(scores["semantic"], iso3) is not None and rank_of(scores["semantic"], iso3) > 50 and
            rank_of(scores["multivec"], iso3) is not None and rank_of(scores["multivec"], iso3) > 50 and
            rank_of(scores["hybrid"],   iso3) is not None and rank_of(scores["hybrid"],   iso3) > 50
            for iso3 in in_dataset
        )
        if all_miss:
            causes.add("SEMANTIC_MISS")

    # BROAD_QUERY: top20 scores very compressed
    top20_sem_scores = [s for _,_,s in top20_sem[:20]]
    if len(top20_sem_scores) >= 20 and (max(top20_sem_scores) - min(top20_sem_scores)) < 0.04:
        causes.add("BROAD_QUERY")

    return sorted(causes)


# ── per-query report ──────────────────────────────────────────────────────────
def hits_at_k(scores_arr, expected_iso3, k):
    in_dataset = [iso3 for iso3 in expected_iso3 if iso3 in _ISO3_TO_IDX]
    if not in_dataset:
        return 0, 0
    hits = sum(1 for iso3 in in_dataset if rank_of(scores_arr, iso3) <= k)
    return hits, len(in_dataset)


def run_query(query_text, category, expected_iso3):
    vec = embed(query_text)
    scores = compute_scores(vec, query_text)

    t20_sem = top_k(scores["semantic"], 20)
    t20_mv  = top_k(scores["multivec"], 20)
    t20_hyb = top_k(scores["hybrid"],   20)

    # hits @ rank ≤ 20
    sem_hits, total = hits_at_k(scores["semantic"], expected_iso3, 20)
    mv_hits,  _     = hits_at_k(scores["multivec"], expected_iso3, 20)
    hyb_hits, _     = hits_at_k(scores["hybrid"],   expected_iso3, 20)

    # per-expected country ranks in each mode
    rank_rows = []
    for iso3 in expected_iso3:
        if iso3 not in _ISO3_TO_IDX:
            rank_rows.append((iso3, _ISO3_TO_NAME.get(iso3,"?"), None, None, None, "NOT_IN_DATASET"))
            continue
        rs = rank_of(scores["semantic"], iso3)
        rm = rank_of(scores["multivec"], iso3)
        rh = rank_of(scores["hybrid"],   iso3)
        rank_rows.append((iso3, _ISO3_TO_NAME.get(iso3, iso3), rs, rm, rh, ""))

    # small-territory surprises in top10
    surprises = []
    for iso3, name, sc in t20_sem[:10]:
        if iso3 not in _ISO3_TO_IDX:
            continue
        idx = _ISO3_TO_IDX[iso3]
        if PROFILE_LENS[idx] < SMALL_TERRITORY_THRESHOLD and iso3 not in expected_iso3:
            surprises.append(f"{name}({iso3},p={PROFILE_LENS[idx]})")
    for iso3, name, sc in t20_hyb[:10]:
        if iso3 not in _ISO3_TO_IDX:
            continue
        idx = _ISO3_TO_IDX[iso3]
        if PROFILE_LENS[idx] < SMALL_TERRITORY_THRESHOLD and iso3 not in expected_iso3:
            if f"{name}({iso3})" not in " ".join(surprises):
                surprises.append(f"{name}({iso3},p={PROFILE_LENS[idx]})")

    causes = flag_causes(t20_sem, t20_mv, t20_hyb, scores, expected_iso3, category)

    best = "semantic"
    if mv_hits > sem_hits and mv_hits >= hyb_hits:
        best = "multivec"
    elif hyb_hits > sem_hits and hyb_hits >= mv_hits:
        best = "hybrid"

    return {
        "query": query_text, "category": category, "expected": expected_iso3,
        "t10_sem": t20_sem[:10], "t10_mv": t20_mv[:10], "t10_hyb": t20_hyb[:10],
        "sem_hits": sem_hits, "mv_hits": mv_hits, "hyb_hits": hyb_hits,
        "total_expected_in_dataset": total,
        "rank_rows": rank_rows, "surprises": surprises, "causes": causes,
        "best": best, "scores": scores,
    }


# ── printing ──────────────────────────────────────────────────────────────────
W = 23

def fmt_entry(iso3, name, score):
    label = f"{name[:13]}({iso3})"
    return f"{label:<18} {score:.3f}"


def print_result(r, idx):
    q = r["query"]
    cat = r["category"]
    total = r["total_expected_in_dataset"]
    sem_h, mv_h, hyb_h = r["sem_hits"], r["mv_hits"], r["hyb_hits"]

    print(f"\n{'='*90}")
    print(f"[{idx:03d}] [{cat}] \"{q}\"")
    print(f"  Hits@20:  semantic={sem_h}/{total}  multivec={mv_h}/{total}  hybrid={hyb_h}/{total}  BEST={r['best'].upper()}")
    print(f"  {'Rank':<4}  {'SEMANTIC':<{W}}  {'MULTIVEC':<{W}}  {'HYBRID':<{W}}")
    print(f"  {'----':<4}  {'-'*W}  {'-'*W}  {'-'*W}")
    for i, (s, m, h) in enumerate(zip(r["t10_sem"], r["t10_mv"], r["t10_hyb"]), 1):
        print(f"  {i:<4}  {fmt_entry(*s):<{W}}  {fmt_entry(*m):<{W}}  {fmt_entry(*h):<{W}}")

    # per-expected rank table (only show if not trivially good)
    tricky = [(iso3,nm,rs,rm,rh,note) for iso3,nm,rs,rm,rh,note in r["rank_rows"]
              if note or (rs and rs > 20) or (rm and rm > 20) or (rh and rh > 20)]
    if tricky:
        print(f"\n  Expected countries with rank > 20 (or not in dataset):")
        print(f"  {'ISO3':<6} {'Name':<22} {'Sem':>5} {'MV':>5} {'Hyb':>5} {'Note'}")
        for iso3,nm,rs,rm,rh,note in sorted(tricky, key=lambda x: x[2] or 999):
            print(f"  {iso3:<6} {nm[:20]:<22} {str(rs or '-'):>5} {str(rm or '-'):>5} {str(rh or '-'):>5} {note}")

    if r["surprises"]:
        print(f"\n  SURPRISES in top10: {', '.join(r['surprises'])}")
    if r["causes"]:
        print(f"  ROOT CAUSES: {', '.join(r['causes'])}")


# ── summary report ────────────────────────────────────────────────────────────
def print_summary(results):
    print(f"\n\n{'#'*90}")
    print(f"  SUMMARY — {len(results)} queries")
    print(f"{'#'*90}")

    total_exp = sum(r["total_expected_in_dataset"] for r in results)
    total_sem = sum(r["sem_hits"] for r in results)
    total_mv  = sum(r["mv_hits"]  for r in results)
    total_hyb = sum(r["hyb_hits"] for r in results)
    print(f"\n  Overall hits@20 across all queries:")
    print(f"    Semantic : {total_sem}/{total_exp} ({100*total_sem/total_exp:.1f}%)")
    print(f"    Multivec : {total_mv}/{total_exp}  ({100*total_mv/total_exp:.1f}%)")
    print(f"    Hybrid   : {total_hyb}/{total_exp} ({100*total_hyb/total_exp:.1f}%)")

    # best mode distribution
    best_counts = {}
    for r in results:
        best_counts[r["best"]] = best_counts.get(r["best"], 0) + 1
    print(f"\n  Best mode per query: {best_counts}")

    # multivec vs semantic delta
    mv_better = sum(1 for r in results if r["mv_hits"] > r["sem_hits"])
    mv_worse  = sum(1 for r in results if r["mv_hits"] < r["sem_hits"])
    hyb_better = sum(1 for r in results if r["hyb_hits"] > r["sem_hits"])
    hyb_worse  = sum(1 for r in results if r["hyb_hits"] < r["sem_hits"])
    print(f"\n  Multivec vs Semantic: better={mv_better} queries, worse={mv_worse} queries")
    print(f"  Hybrid   vs Semantic: better={hyb_better} queries, worse={hyb_worse} queries")

    # root cause frequency
    from collections import Counter
    all_causes = []
    for r in results:
        all_causes.extend(r["causes"])
    cause_counts = Counter(all_causes)
    print(f"\n  Root cause frequency:")
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        pct = 100*cnt/len(results)
        bar = "█" * int(pct/3)
        print(f"    {cause:<45} {cnt:>3} queries ({pct:5.1f}%)  {bar}")

    # per-category summary
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"sem":0,"mv":0,"hyb":0,"total":0,"n":0})
    for r in results:
        c = r["category"]
        by_cat[c]["sem"]   += r["sem_hits"]
        by_cat[c]["mv"]    += r["mv_hits"]
        by_cat[c]["hyb"]   += r["hyb_hits"]
        by_cat[c]["total"] += r["total_expected_in_dataset"]
        by_cat[c]["n"]     += 1

    print(f"\n  Per-category hits@20 (sem / mv / hyb of total expected):")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        t = d["total"] or 1
        print(f"    {cat:<16} n={d['n']}  sem={d['sem']}/{t}({100*d['sem']/t:.0f}%)  mv={d['mv']}/{t}({100*d['mv']/t:.0f}%)  hyb={d['hyb']}/{t}({100*d['hyb']/t:.0f}%)")

    # worst 10 queries (by semantic hits)
    worst = sorted(results, key=lambda r: r["sem_hits"] / max(r["total_expected_in_dataset"],1))[:10]
    print(f"\n  10 worst queries (lowest semantic hit rate):")
    for r in worst:
        t = r["total_expected_in_dataset"] or 1
        print(f"    sem={r['sem_hits']}/{t}  [{r['category']}] \"{r['query']}\"  causes={r['causes']}")

    # queries where hybrid DEGRADES vs semantic
    degraded = [r for r in results if r["hyb_hits"] < r["sem_hits"]]
    if degraded:
        print(f"\n  Queries where hybrid DEGRADES vs semantic ({len(degraded)}):")
        for r in sorted(degraded, key=lambda r: r["sem_hits"]-r["hyb_hits"], reverse=True):
            print(f"    sem={r['sem_hits']}→hyb={r['hyb_hits']}  [{r['category']}] \"{r['query']}\"")

    # queries where multivec DEGRADES vs semantic
    mv_degraded = [r for r in results if r["mv_hits"] < r["sem_hits"]]
    if mv_degraded:
        print(f"\n  Queries where multivec DEGRADES vs semantic ({len(mv_degraded)}):")
        for r in sorted(mv_degraded, key=lambda r: r["sem_hits"]-r["mv_hits"], reverse=True):
            print(f"    sem={r['sem_hits']}→mv={r['mv_hits']}  [{r['category']}] \"{r['query']}\"")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Audit: {len(QUERIES)} queries  |  {N_ENTITIES} countries  |  3 modes")
    print(f"ISO3 codes not in dataset will be skipped in hit counting.\n")

    # warn about ISO3 codes not in dataset
    all_expected = set()
    for _, _, exp in QUERIES:
        all_expected.update(exp)
    missing = sorted(all_expected - _ISO3_SET)
    if missing:
        print(f"ISO3 codes in expected lists but not in dataset ({len(missing)}): {missing}\n")

    results = []
    for i, (query_text, category, expected_iso3) in enumerate(QUERIES, 1):
        print(f"  [{i:03d}/{len(QUERIES)}] {category}: {query_text[:60]}", flush=True)
        r = run_query(query_text, category, expected_iso3)
        results.append(r)

    # detailed per-query output
    print("\n\n" + "="*90)
    print("  DETAILED RESULTS")
    print("="*90)
    for i, r in enumerate(results, 1):
        print_result(r, i)

    # summary
    print_summary(results)

    print(f"\n\nAudit complete. Results in audit_results.txt if you piped output.")


if __name__ == "__main__":
    main()
