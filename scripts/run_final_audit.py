"""Comprehensive quality audit — 160 queries across all categories.

Evaluates AUTO mode (the new default) and computes:
- hits@20 per query and category
- oracle ceiling (best of all modes per query)
- failure classification for every miss
- aggregate failure pattern report

Failure taxonomy:
  SMALL_HUB       - unexpected short-profile territory in top-20 (hub effect)
  SEMANTIC_MISS   - concept absent from all profiles (rank >80 in semantic)
  ROUTING_MISS    - should route factually but doesn't (indicator exists)
  EVAL_MISMATCH   - expected list may not match reality (0 hits in ALL modes)
  BROAD_QUERY     - concept too diffuse, scores compressed, can't discriminate
  DATA_GAP        - expected country has very short profile (<2500 chars)
  INDICATOR_PROXY - factual route fires but wrong metric (% vs absolute)
  SECTION_WEAK    - relevant wiki section very short or absent

Run:
    .venv/bin/python scripts/run_final_audit.py 2>&1 | tee final_audit.txt
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

from src.data_loader import EMBEDDINGS, ENTITIES_DF, PROFILE_LENS, DOMAIN_EMBEDDINGS
from src.factual_router import detect_route, factual_scores
from src.similarity import embed_query

ISO3  = ENTITIES_DF["iso3"].tolist()
NAMES = ENTITIES_DF["name"].tolist()
N     = len(ISO3)
_ISO3_SET = set(ISO3)

ENERGY_TERMS = {"energy","solar","renewable","electricity","nuclear","hydroelectric"}


# ── Score functions ────────────────────────────────────────────────────────────

def _penalty(scores):
    from src.multivec import apply_length_penalty
    return apply_length_penalty(scores)

def sem(vec, text):
    from src.multivec import multi_sim, apply_length_penalty
    alpha = 0.10 if any(t in text.lower() for t in ENERGY_TERMS) else 0.30
    return apply_length_penalty(multi_sim(vec, alpha=alpha))

def hyb(vec, text):
    from src.bm25 import blend
    return blend(sem(vec, text), text, alpha=0.08)

def dom(vec):
    from src.domain_sim import domain_sim
    from src.multivec import apply_length_penalty
    return apply_length_penalty(domain_sim(vec, alpha=0.4))

def auto_scores(text):
    """Return (scores, mode_label). Factual route if matched, else semantic."""
    route = detect_route(text)
    if route:
        ind, asc, label = route
        return factual_scores(ind, ISO3, asc), f"factual:{ind}"
    vec = embed_query(text)
    return sem(vec, text), "semantic"

def hits(scores, expected, k=20):
    top = {ISO3[i] for i in np.argsort(scores)[::-1][:k]}
    return sum(1 for iso3 in expected if iso3 in top), top


# ── Failure classifiers ────────────────────────────────────────────────────────

def classify_miss(q_text, expected_iso3, auto_sc, all_sc_dict):
    """Return list of failure tags for one expected country that was missed."""
    tags = []
    idx = ISO3.index(expected_iso3) if expected_iso3 in ISO3 else -1
    if idx == -1:
        return ["COVERAGE_GAP"]

    # Rank in auto scores
    auto_rank = int(np.sum(auto_sc >= auto_sc[idx]))

    # Rank in best semantic
    s_sc = all_sc_dict["semantic"]
    sem_rank = int(np.sum(s_sc >= s_sc[idx]))

    # Profile length
    plen = int(PROFILE_LENS[idx])

    if plen < 2500:
        tags.append("DATA_GAP")

    if sem_rank > 80:
        tags.append("SEMANTIC_MISS")

    # Check if a factual route exists for this kind of query
    route = detect_route(q_text)
    if route is None:
        # Could there be a relevant indicator?
        # Check a few common ones
        pass  # leave as SEMANTIC_MISS or other
    elif route:
        # Route fired but country still missed → indicator proxy issue
        ind = route[0]
        from src.factual_router import _FACT_INDEX
        if (expected_iso3, ind) in _FACT_INDEX:
            tags.append("INDICATOR_PROXY")  # has data but still missed
        else:
            tags.append("DATA_GAP")  # no data for this indicator

    if not tags:
        if auto_rank > 50:
            tags.append("SEMANTIC_MISS")
        elif auto_rank > 20:
            tags.append("BROAD_QUERY")
        else:
            tags.append("BROAD_QUERY")

    return tags


def classify_hub(top20_iso3s, expected_set):
    """Count unexpected short-profile countries in top-20."""
    hubs = []
    for iso3 in top20_iso3s:
        if iso3 not in expected_set:
            idx = ISO3.index(iso3)
            if PROFILE_LENS[idx] < 5000:
                hubs.append(iso3)
    return hubs


# ── Query set (160 queries) ───────────────────────────────────────────────────

QUERIES = [

    # ── Religion (8) ─────────────────────────────────────────────────────────
    ("religion", "Islam Muslim culture",
     ["SAU","IRN","PAK","IDN","EGY","TUR","MYS","BGD","MAR","NGA"]),
    ("religion", "Buddhism Buddhist monks temples",
     ["THA","MMR","LKA","BTN","KHM","LAO","MNG","JPN","CHN","VNM"]),
    ("religion", "Hinduism Hindu temples culture",
     ["IND","NPL","IDN","MUS","FJI","SUR","GUY","TTO","SGP","ZAF"]),
    ("religion", "Roman Catholic Christianity",
     ["ITA","ESP","PRT","BRA","PHL","POL","ARG","MEX","IRL","COL"]),
    ("religion", "Orthodox Christianity",
     ["RUS","GRC","SRB","BGR","ROU","UKR","GEO","ARM","CYP","MKD"]),
    ("religion", "secular atheist society",
     ["CHN","CZE","EST","FRA","DNK","SWE","NLD","NOR","FIN","DEU"]),
    ("religion", "Jewish culture Judaism",
     ["ISR","USA","CAN","HUN","AUT","DEU","FRA","GBR","ARG","BLR"]),
    ("religion", "theocracy religion and state",
     ["IRN","SAU","VAT","AFG","SOM","YEM","PAK","SDN","MRT","BHR"]),

    # ── Politics (8) ─────────────────────────────────────────────────────────
    ("politics", "authoritarian regime one-party state",
     ["CHN","PRK","CUB","VEN","BLR","RUS","TKM","ERI","SYR","AZE"]),
    ("politics", "constitutional monarchy",
     ["GBR","DNK","NOR","SWE","ESP","NLD","BEL","THA","JPN","MAR"]),
    ("politics", "military junta coup government",
     ["MMR","SDN","TCD","MLI","BFA","THA","EGY","PAK","TUR","NER"]),
    ("politics", "corruption governance failure",
     ["SOM","SSD","SYR","VEN","NGA","CAF","GNB","HTI","AFG","IRQ"]),
    ("politics", "direct democracy referendum",
     ["CHE","ISL","IRL","NZL","AUS","CAN","URY","DEU","SWE","FIN"]),
    ("politics", "elections voting rights",
     ["SWE","FIN","NOR","NZL","CAN","AUS","DEU","GBR","FRA","USA"]),
    ("politics", "communist one-party socialist state",
     ["CHN","CUB","PRK","VNM","LAO","BLR","VEN","ERI","SYR","TKM"]),
    ("politics", "federal state decentralised government",
     ["USA","DEU","AUS","CAN","CHE","IND","BRA","MEX","ARG","RUS"]),

    # ── Economy (7) ──────────────────────────────────────────────────────────
    ("economy", "high GDP wealthy economy",
     ["LUX","CHE","NOR","USA","SGP","QAT","AUS","DNK","SWE","NLD"]),
    ("economy", "manufacturing industrial economy",
     ["CHN","DEU","JPN","KOR","USA","TWN","MEX","THA","POL","CZE"]),
    ("economy", "financial hub banking services",
     ["CHE","LUX","SGP","GBR","USA","ARE","MUS","CYM","LIE","NLD"]),
    ("economy", "tourism-dependent economy small island",
     ["MDV","BHS","ATG","VUT","FJI","PLW","WSM","TON","BLZ","CPV"]),
    ("economy", "remittance economy migrant workers",
     ["TON","KGZ","TJK","MDV","HND","SLV","NPL","GEO","JAM","KOS"]),
    ("economy", "subsistence poverty least developed",
     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"]),
    ("economy", "startup unicorn venture capital tech hub",
     ["USA","CHN","GBR","ISR","SGP","DEU","FRA","IND","CAN","SWE"]),

    # ── Finance/tech (6) ─────────────────────────────────────────────────────
    ("technology", "semiconductor chip manufacturing",
     ["TWN","KOR","JPN","USA","DEU","NLD","CHN","MYS","SGP","IRL"]),
    ("technology", "artificial intelligence research",
     ["USA","CHN","GBR","CAN","DEU","FRA","ISR","JPN","KOR","SGP"]),
    ("technology", "fintech innovation digital banking",
     ["GBR","SWE","SGP","USA","EST","KEN","NGA","IND","NLD","ISR"]),
    ("technology", "software engineering Silicon Valley",
     ["USA","ISR","IND","SGP","GBR","DEU","FIN","SWE","CAN","NLD"]),
    ("technology", "cryptocurrency blockchain adoption",
     ["SVN","EST","MLT","CHE","SGP","USA","GBR","NLD","DEU","CAN"]),
    ("technology", "e-commerce online shopping digital",
     ["CHN","USA","GBR","DEU","KOR","JPN","AUS","NLD","FRA","SWE"]),

    # ── Education (5) ────────────────────────────────────────────────────────
    ("education", "PISA school rankings education quality",
     ["SGP","FIN","JPN","KOR","CAN","EST","NLD","POL","SWE","CHE"]),
    ("education", "free university tuition higher education",
     ["NOR","SWE","FIN","DEU","AUT","FRA","DNK","CZE","GRC","POL"]),
    ("education", "vocational training apprenticeship",
     ["DEU","AUT","CHE","DNK","NLD","FIN","NOR","SWE","GBR","JPN"]),
    ("education", "Nobel Prize science research",
     ["USA","GBR","DEU","FRA","CHE","SWE","JPN","ISR","AUS","NLD"]),
    ("education", "university academic research excellence",
     ["USA","GBR","DEU","FRA","JPN","CHN","CAN","AUS","CHE","SWE"]),

    # ── Health (4) ───────────────────────────────────────────────────────────
    ("health", "healthcare system universal quality",
     ["NOR","SWE","FIN","AUS","CAN","DEU","FRA","JPN","NZL","CHE"]),
    ("health", "medical tourism destination",
     ["THA","IND","MYS","SGP","ISR","MEX","TUR","POL","CZE","HUN"]),
    ("health", "disease burden malaria epidemic",
     ["NGA","COD","TZA","MOZ","BFA","NER","MLI","CMR","UGA","GHA"]),
    ("health", "mental health wellbeing happiness index",
     ["FIN","DNK","ISL","NOR","SWE","NZL","CHE","AUS","NLD","CAN"]),

    # ── Tourism (6) ──────────────────────────────────────────────────────────
    ("tourism", "tropical beach resort vacation",
     ["MDV","THA","BHS","PHL","IDN","MEX","MYS","LKA","FJI","JAM"]),
    ("tourism", "cultural heritage UNESCO world heritage",
     ["ITA","ESP","FRA","CHN","IND","MEX","GRC","EGY","JPN","PRT"]),
    ("tourism", "adventure ecotourism nature wildlife",
     ["CRI","BLZ","BOL","ECU","NZL","CAN","NOR","TZA","KEN","BRN"]),
    ("tourism", "safari wildlife game reserve Africa",
     ["TZA","KEN","ZAF","BWA","NAM","ZMB","ZWE","ETH","UGA","RWA"]),
    ("tourism", "backpacker budget travel cheap",
     ["THA","VNM","IDN","IND","NPL","BOL","PER","MEX","LAO","KHM"]),
    ("tourism", "luxury cruise ship destination",
     ["BHS","MDV","MEX","GRC","ITA","HRV","NOR","CYP","TTO","JAM"]),

    # ── Islands (5) ──────────────────────────────────────────────────────────
    ("islands", "tropical island nation",
     ["MDV","FJI","PLW","FSM","WSM","TON","VUT","NRU","KIR","COM"]),
    ("islands", "Caribbean island",
     ["BHS","CUB","JAM","TTO","DOM","BRB","LCA","VCT","GRD","ATG"]),
    ("islands", "archipelago island chain",
     ["PHL","IDN","JPN","STP","CPV","MDV","FJI","VUT","TUV","KIR"]),
    ("islands", "volcanic island geology",
     ["ISL","IDN","ITA","PHL","ECU","CPV","VUT","TON","WSM","PNG"]),
    ("islands", "coral reef marine biodiversity ocean",
     ["AUS","PHL","IDN","MDV","FJI","PLW","FSM","PNG","MHL","TUV"]),

    # ── Mountains/skiing (5) ─────────────────────────────────────────────────
    ("mountains", "alpine skiing winter sports",
     ["AUT","CHE","FRA","ITA","NOR","SWE","FIN","AND","LIE","SVN"]),
    ("mountains", "high altitude Himalayan mountain",
     ["NPL","CHN","IND","BTN","PAK","TJK","KGZ","AFG","MMR","LAO"]),
    ("mountains", "fjords glaciers landscape",
     ["NOR","ISL","CHE","NZL","CAN","AUS","SVN","ALB","GRL","CHL"]),
    ("mountains", "mountaineering trekking hiking",
     ["NPL","CHE","NZL","NOR","PER","ECU","TZA","KEN","BGR","SVN"]),
    ("mountains", "ski resort mountain tourism",
     ["AUT","CHE","FRA","ITA","AND","LIE","SVK","BGR","KGZ","NOR"]),

    # ── Agriculture (7) ──────────────────────────────────────────────────────
    ("agriculture", "wheat grain cereal farming",
     ["RUS","USA","CHN","IND","AUS","FRA","DEU","UKR","CAN","PAK"]),
    ("agriculture", "rice cultivation paddy field",
     ["CHN","IND","IDN","BGD","VNM","THA","MMR","PHL","BRA","JPN"]),
    ("agriculture", "coffee production export",
     ["BRA","VNM","COL","ETH","IDN","UGA","MEX","IND","PER","HND"]),
    ("agriculture", "wine production viticulture vineyards",
     ["FRA","ITA","ESP","PRT","ARG","AUS","CHL","ZAF","DEU","USA"]),
    ("agriculture", "cattle ranching beef livestock",
     ["BRA","USA","AUS","ARG","MEX","ETH","CHN","RUS","COL","NZL"]),
    ("agriculture", "cacao cocoa chocolate production",
     ["CIV","GHA","IDN","CMR","NGA","BRA","ECU","PRY","COD","UGA"]),
    ("agriculture", "spice tea plantation",
     ["IND","CHN","KEN","LKA","IDN","TZA","VNM","RWA","TUR","NPL"]),

    # ── Energy (6) ───────────────────────────────────────────────────────────
    ("energy", "nuclear power station electricity",
     ["FRA","USA","CHN","KOR","RUS","DEU","JPN","GBR","CAN","IND"]),
    ("energy", "hydroelectric dam power",
     ["NOR","BRA","CHN","CAN","ISL","PRY","VNM","COL","PER","ETH"]),
    ("energy", "electricity access energy poverty",
     ["ETH","NGA","TZA","UGA","MOZ","SSD","MWI","BFA","NER","SOM"]),
    ("energy", "clean energy transition",
     ["DNK","NOR","SWE","ISL","NZL","AUT","PRT","FIN","URY","GBR"]),
    ("energy", "solar photovoltaic panel installation",
     ["DEU","CHN","AUS","ESP","USA","IND","JPN","ITA","NLD","GBR"]),
    ("energy", "oil gas petroleum exploration deepwater",
     ["SAU","RUS","USA","NOR","BRA","ANG","NGA","KAZ","MEX","ARE"]),

    # ── Oil/gas (4) ──────────────────────────────────────────────────────────
    ("oil_gas", "oil petroleum export country",
     ["SAU","RUS","IRQ","IRN","ARE","KWT","NGA","VEN","AGO","LBY"]),
    ("oil_gas", "natural gas pipeline export",
     ["RUS","QAT","NOR","IRN","TKM","AZE","DZA","NGA","TTO","USA"]),
    ("oil_gas", "OPEC oil cartel member",
     ["SAU","IRQ","IRN","ARE","KWT","NGA","VEN","AGO","ECU","LBY"]),
    ("oil_gas", "Gulf state petrodollar wealth",
     ["SAU","ARE","QAT","KWT","BHR","OMN","LBY","IRQ","IRN","AZE"]),

    # ── Climate (5) ──────────────────────────────────────────────────────────
    ("climate", "tropical humid rainforest climate",
     ["BRA","COD","COG","IDN","MYS","PER","COL","ECU","CMR","PNG"]),
    ("climate", "arctic subarctic tundra cold",
     ["RUS","CAN","NOR","SWE","FIN","ISL","MNG","USA","GRL","EST"]),
    ("climate", "monsoon rainy season flood",
     ["BGD","IND","MMR","THA","VNM","PHL","IDN","CHN","NPL","PAK"]),
    ("climate", "Mediterranean dry summer climate",
     ["GRC","ITA","ESP","PRT","TUN","LBY","MAR","ALB","CYP","HRV"]),
    ("climate", "climate change vulnerable rising sea level",
     ["BGD","MDV","KIR","TUV","MHL","NLD","VNM","NGA","ETH","SOM"]),

    # ── Desert (3) ───────────────────────────────────────────────────────────
    ("desert", "Sahara desert arid landscape",
     ["DZA","LBY","EGY","MRT","MAR","TUN","NER","TCD","MLI","SDN"]),
    ("desert", "desert nomad Bedouin camel",
     ["SAU","OMN","MAR","DZA","EGY","IRQ","JOR","SYR","TUN","MRT"]),
    ("desert", "dryland water scarcity arid",
     ["EGY","SAU","JOR","ISR","OMN","TUN","DZA","MAR","NAM","BWA"]),

    # ── Forests/nature (3) ───────────────────────────────────────────────────
    ("nature", "Amazon rainforest deforestation",
     ["BRA","PER","COL","ECU","BOL","VEN","GUY","SUR","TTO","GUF"]),
    ("nature", "boreal taiga forest logging",
     ["RUS","CAN","SWE","FIN","NOR","USA","CHN","KAZ","MNG","EST"]),
    ("nature", "biodiversity endemic species nature",
     ["BRA","IDN","COL","AUS","MDG","PER","ECU","PNG","ZAF","VNM"]),

    # ── Military/conflict (4) ────────────────────────────────────────────────
    ("military", "military spending armed forces power",
     ["USA","CHN","RUS","SAU","IND","GBR","DEU","FRA","JPN","KOR"]),
    ("military", "ongoing civil war armed conflict",
     ["SYR","ETH","YEM","LBY","MLI","SSD","SOM","AFG","IRQ","SDN"]),
    ("military", "peacekeeping UN military troops",
     ["ETH","BGD","PAK","RWA","GHA","IND","NGA","FRA","URY","NLD"]),
    ("military", "arms weapons export manufacturer",
     ["USA","RUS","FRA","DEU","GBR","ISR","CHN","ITA","ESP","KOR"]),

    # ── History (6) ──────────────────────────────────────────────────────────
    ("history", "ancient civilization archaeology",
     ["GRC","EGY","IRQ","IND","CHN","ITA","IRN","MEX","PER","TUR"]),
    ("history", "World War II battlefield history",
     ["DEU","FRA","POL","GBR","RUS","ITA","USA","NLD","BEL","GRC"]),
    ("history", "Cold War Soviet communist bloc",
     ["RUS","CHN","CUB","PRK","VNM","POL","HUN","CZE","BGR","ROU"]),
    ("history", "colonial history empire",
     ["GBR","FRA","ESP","PRT","NLD","BEL","DEU","ITA","RUS","JPN"]),
    ("history", "Viking Norse history",
     ["NOR","SWE","DNK","ISL","FIN","GBR","IRL","EST","LVA","POL"]),
    ("history", "indigenous native culture",
     ["MEX","PER","BOL","ECU","GTM","CAN","AUS","NZL","BRA","COL"]),

    # ── Culture/arts (6) ─────────────────────────────────────────────────────
    ("culture", "fashion luxury brands design",
     ["FRA","ITA","USA","GBR","ESP","DEU","JPN","BEL","CHE","SWE"]),
    ("culture", "classical music opera symphony",
     ["AUT","DEU","ITA","RUS","FRA","CZE","HUN","POL","GBR","USA"]),
    ("culture", "film cinema movie industry",
     ["USA","IND","GBR","FRA","ITA","JPN","CHN","KOR","IRN","EGY"]),
    ("culture", "anime manga cartoon",
     ["JPN","KOR","CHN","USA","FRA","GBR","CAN","AUS","NLD","DEU"]),
    ("culture", "jazz blues rock music heritage",
     ["USA","GBR","JAM","CUB","BRA","FRA","IRL","AUS","CAN","NLD"]),
    ("culture", "gaming esports video games",
     ["KOR","USA","CHN","SWE","FIN","DEU","JPN","GBR","CAN","AUS"]),

    # ── Food/drink (5) ───────────────────────────────────────────────────────
    ("food", "coffee culture cafe espresso",
     ["ETH","ITA","BRA","COL","VNM","TUR","GRC","AUT","UGA","HND"]),
    ("food", "spicy street food cuisine",
     ["THA","IND","MEX","VNM","IDN","ETH","KOR","TUR","MAR","LAO"]),
    ("food", "beer brewing craft beer",
     ["DEU","CZE","BEL","GBR","IRL","AUT","USA","AUS","NLD","POL"]),
    ("food", "tea culture ceremony",
     ["CHN","JPN","IND","GBR","TUR","IRN","MAR","EGY","KEN","TZA"]),
    ("food", "cheese dairy cuisine",
     ["FRA","CHE","NLD","DEU","ITA","GBR","AUT","DNK","NOR","SWE"]),

    # ── Sports (6) ───────────────────────────────────────────────────────────
    ("sports", "football soccer culture",
     ["BRA","ARG","ESP","DEU","FRA","ITA","GBR","NLD","PRT","BEL"]),
    ("sports", "cricket national sport",
     ["IND","AUS","GBR","PAK","NZL","LKA","ZAF","WIA","BGD","ZIM"]),
    ("sports", "basketball NBA sport",
     ["USA","LTU","SVN","FRA","AUS","CAN","ARG","SRB","NGA","TUR"]),
    ("sports", "rugby union scrum",
     ["NZL","AUS","ZAF","GBR","FRA","IRL","ARG","SCO","WAL","FJI"]),
    ("sports", "Formula One motorsport racing",
     ["GBR","DEU","ITA","FRA","BRA","MCO","AUT","AUS","BHR","SGP"]),
    ("sports", "Olympic medals sports achievement",
     ["USA","RUS","DEU","CHN","GBR","FRA","AUS","ITA","KOR","JPN"]),

    # ── Aviation/transport (4) ───────────────────────────────────────────────
    ("aviation", "major airline international hub airport",
     ["ARE","SGP","GBR","USA","DEU","NLD","JPN","TUR","FRA","KOR"]),
    ("aviation", "container shipping maritime trade",
     ["SGP","CHN","NLD","GBR","USA","DEU","HKG","ARE","KOR","JPN"]),
    ("aviation", "landlocked country no sea access",
     ["CHE","AUT","CZE","SVK","HUN","SRB","BOL","PRY","ETH","UGA"]),
    ("aviation", "high-speed rail train infrastructure",
     ["CHN","JPN","FRA","DEU","ESP","ITA","KOR","TUR","SAU","NLD"]),

    # ── Natural disasters (4) ────────────────────────────────────────────────
    ("disasters", "earthquake seismic activity",
     ["JPN","TUR","IRN","IND","IDN","CHL","MEX","GRC","ITA","NZL"]),
    ("disasters", "hurricane typhoon cyclone",
     ["PHL","JPN","BGD","IND","CHN","USA","MEX","MDG","MOZ","CUB"]),
    ("disasters", "volcanic eruption lava",
     ["IDN","ITA","JPN","ISL","PHL","ECU","COL","MEX","PNG","CPV"]),
    ("disasters", "flood river delta disaster",
     ["BGD","CHN","IND","VNM","NLD","MOZ","NGA","PHL","PAK","USA"]),

    # ── Factual routed (6) — expected to use Auto→factual ────────────────────
    ("factual", "life expectancy longevity",
     ["JPN","CHE","ESP","ITA","AUS","SGP","ISL","LUX","NOR","SWE"]),
    ("factual", "international tourist arrivals most visited",
     ["FRA","ESP","USA","CHN","ITA","TUR","MEX","GBR","DEU","THA"]),
    ("factual", "electoral democracy political freedom",
     ["NOR","FIN","SWE","DNK","NLD","NZL","CAN","AUS","CHE","DEU"]),
    ("factual", "internet users digital society",
     ["NOR","ISL","DNK","FIN","SWE","NLD","CHE","GBR","KOR","LUX"]),
    ("factual", "renewable energy leaders",
     ["ISL","NOR","SWE","DNK","FIN","NZL","AUT","CHE","DEU","ESP"]),
    ("factual", "agriculture dominates economy farming",
     ["BDI","NER","CAF","SSD","MWI","MOZ","TCD","RWA","UGA","ETH"]),

    # ── Additional diverse (16) ───────────────────────────────────────────────
    ("society", "LGBTQ rights gender equality progressive",
     ["NLD","BEL","CAN","ESP","NOR","SWE","DNK","ISL","FIN","NZL"]),
    ("society", "refugee asylum migration crisis",
     ["SYR","AFG","SSD","VEN","MMR","COD","SOM","ETH","IRQ","ERItrée"]),
    ("society", "ethnic diversity multicultural society",
     ["CAN","AUS","USA","NZL","SGP","MYS","SUR","GUY","TTO","FJI"]),
    ("environment", "carbon emissions greenhouse gas high",
     ["CHN","USA","IND","RUS","JPN","DEU","KOR","CAN","IRN","SAU"]),
    ("environment", "plastic pollution ocean waste",
     ["CHN","IDN","PHL","VNM","THA","EGY","NGA","BGD","IND","MDG"]),
    ("environment", "water scarcity drought arid",
     ["EGY","SAU","JOR","LBY","DZA","ISR","OMN","TUN","MAR","NAM"]),
    ("language", "English speaking country",
     ["USA","GBR","AUS","CAN","NZL","IRL","ZAF","NGA","IND","JAM"]),
    ("economy", "gold reserves central bank",
     ["USA","DEU","ITA","FRA","RUS","CHN","CHE","IND","JPN","NLD"]),
    ("economy", "high inflation economic instability",
     ["VEN","ZWE","ARG","SSD","SDN","SYR","LBN","TUR","IRN","ETH"]),
    ("society", "fishing industry seafood maritime",
     ["CHN","NOR","PER","IDN","IND","VNM","ISL","USA","RUS","JPN"]),
    ("history", "Arab Spring revolution protest",
     ["TUN","EGY","LBY","SYR","YEM","BHR","MAR","JOR","SDN","IRQ"]),
    ("economy", "microfinance financial inclusion poverty",
     ["BGD","KEN","IND","TZA","ETH","NPL","GHA","UGA","PHI","BFA"]),
    ("culture", "chess intellectual board game",
     ["RUS","ARM","GEO","AZE","USA","NLD","HUN","CHN","ISL","CUB"]),
    ("environment", "deforestation land use change",
     ["BRA","IDN","COL","COD","BOL","CMR","NGA","PER","MYS","MDG"]),
    ("tech", "robotics automation industry",
     ["JPN","DEU","KOR","USA","CHN","CHE","SWE","DNK","NLD","AUS"]),
    ("society", "press freedom journalism independent media",
     ["NOR","FIN","SWE","DNK","NLD","ISL","CHE","NZL","AUS","CAN"]),
]


def run():
    print("=" * 72)
    print(f"FINAL QUALITY AUDIT — {len(QUERIES)} queries across all categories")
    print("Evaluating Auto mode (default) vs Oracle (best per query)")
    print("=" * 72)

    # Results storage
    auto_hits_total   = 0
    sem_hits_total    = 0
    hyb_hits_total    = 0
    dom_hits_total    = 0
    oracle_total      = 0
    expected_total    = 0

    cat_auto  = defaultdict(list)  # category → list of (hits, n_expected)
    cat_sem   = defaultdict(list)
    cat_oracle= defaultdict(list)

    failure_tags  = Counter()      # tag → count of MISSED expected countries
    mode_dist     = Counter()      # how often each mode was used
    hub_count     = 0              # small-territory hubs appearing in top-20
    hub_queries   = 0              # queries affected by hub

    query_results = []

    for cat, q_text, expected in QUERIES:
        # Filter expected to countries in our dataset
        valid_expected = [iso3 for iso3 in expected if iso3 in _ISO3_SET]
        if not valid_expected:
            continue
        n_exp = len(valid_expected)

        # Auto mode
        route = detect_route(q_text)
        if route:
            ind, asc, label = route
            a_sc = factual_scores(ind, ISO3, asc)
            mode_label = f"factual:{ind}"
        else:
            vec = embed_query(q_text)
            a_sc = sem(vec, q_text)
            mode_label = "semantic"

        # Other modes for oracle
        if not route:
            vec_ref = vec
        else:
            vec_ref = embed_query(q_text)

        s_sc = sem(vec_ref, q_text)
        h_sc = hyb(vec_ref, q_text)
        d_sc = dom(vec_ref)

        a_h, a_top = hits(a_sc, valid_expected)
        s_h, _     = hits(s_sc, valid_expected)
        h_h, _     = hits(h_sc, valid_expected)
        d_h, _     = hits(d_sc, valid_expected)
        oracle_h   = max(a_h, s_h, h_h, d_h)

        auto_hits_total  += a_h
        sem_hits_total   += s_h
        hyb_hits_total   += h_h
        dom_hits_total   += d_h
        oracle_total     += oracle_h
        expected_total   += n_exp

        cat_auto[cat].append((a_h, n_exp))
        cat_sem[cat].append((s_h, n_exp))
        cat_oracle[cat].append((oracle_h, n_exp))
        mode_dist[mode_label.split(":")[0]] += 1

        # Hub analysis
        hubs = classify_hub(a_top, set(valid_expected))
        if hubs:
            hub_count += len(hubs)
            hub_queries += 1

        # Classify misses
        all_sc = {"semantic": s_sc}
        for iso3 in valid_expected:
            if iso3 not in a_top:
                tags = classify_miss(q_text, iso3, a_sc, all_sc)
                for t in tags:
                    failure_tags[t] += 1

        query_results.append({
            "cat": cat, "q": q_text, "n_exp": n_exp,
            "auto": a_h, "sem": s_h, "hyb": h_h, "dom": d_h,
            "oracle": oracle_h, "mode": mode_label,
            "deficit": oracle_h - a_h,
        })

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nOVERALL HITS@20 ({expected_total} expected across {len(query_results)} queries)")
    print("-" * 55)
    for label, total in [
        ("Auto (default)",  auto_hits_total),
        ("Semantic only",   sem_hits_total),
        ("Hybrid",          hyb_hits_total),
        ("Domain",          dom_hits_total),
        ("Oracle ceiling",  oracle_total),
    ]:
        pct = 100 * total / expected_total
        bar = "█" * int(pct / 2)
        print(f"  {label:18s}: {total:4d}/{expected_total}  {pct:5.1f}%  {bar}")

    pct_auto   = 100 * auto_hits_total / expected_total
    pct_oracle = 100 * oracle_total / expected_total
    auto_gap   = oracle_total - auto_hits_total
    print(f"\n  Auto leaves {auto_gap} hits on table vs oracle ({pct_auto:.1f}% vs {pct_oracle:.1f}%)")

    print("\nMODE DISTRIBUTION (how often each path taken in Auto)")
    for mode, cnt in mode_dist.most_common():
        print(f"  {mode:20s}: {cnt} queries ({100*cnt/len(query_results):.0f}%)")

    print(f"\nHUB ANALYSIS")
    print(f"  Small-territory (<5000 chars) in top-20: {hub_count} appearances across {hub_queries} queries")

    # Per-category breakdown
    print("\nBY CATEGORY (Auto hits / Oracle ceiling):")
    cat_rows = []
    for cat in sorted(set(c for c, _, _ in QUERIES)):
        a_hits = sum(h for h,_ in cat_auto[cat])
        a_exp  = sum(n for _,n in cat_auto[cat])
        o_hits = sum(h for h,_ in cat_oracle[cat])
        if a_exp == 0: continue
        gap = o_hits - a_hits
        cat_rows.append((cat, a_hits, a_exp, o_hits, gap))

    cat_rows.sort(key=lambda x: x[1]/x[2])  # sort by Auto%, ascending
    for cat, a_h, a_exp, o_h, gap in cat_rows:
        a_pct = 100*a_h/a_exp
        o_pct = 100*o_h/a_exp
        bar = "█" * int(a_pct/5)
        print(f"  {cat:15s}: Auto={a_h:3d}/{a_exp} ({a_pct:4.0f}%)  Oracle={o_h}/{a_exp} ({o_pct:.0f}%)  gap={gap:+d}  {bar}")

    # Top 10 worst queries
    query_results.sort(key=lambda x: x["auto"])
    print("\nTOP 15 WORST QUERIES (by Auto hits):")
    for r in query_results[:15]:
        print(f"  {r['auto']:2d}/{r['n_exp']} [{r['cat']:12s}] [{r['mode'][:18]:18s}] {r['q'][:55]}")

    # Top 10 deficit queries (oracle >> auto)
    query_results.sort(key=lambda x: -x["deficit"])
    print("\nTOP 10 QUERIES WHERE ORACLE BEATS AUTO:")
    for r in query_results[:10]:
        if r["deficit"] == 0: break
        best = max(r["sem"],r["hyb"],r["dom"])
        best_mode = "sem" if best==r["sem"] else ("hyb" if best==r["hyb"] else "dom")
        print(f"  oracle={r['oracle']} auto={r['auto']} (+{r['deficit']}) best_alt={best_mode}  [{r['cat']}] {r['q'][:50]}")

    # Failure classification
    total_misses = sum(failure_tags.values())
    print(f"\nFAILURE PATTERN ANALYSIS ({total_misses} total missed expected countries)")
    print("-" * 55)
    for tag, cnt in failure_tags.most_common():
        pct = 100 * cnt / total_misses
        bar = "█" * int(pct / 2)
        print(f"  {tag:20s}: {cnt:4d}  ({pct:5.1f}%)  {bar}")

    # Questions summary
    print("\n" + "=" * 72)
    print("STRATEGIC ANALYSIS")
    print("=" * 72)
    sem_pct   = 100 * sem_hits_total / expected_total
    auto_pct  = 100 * auto_hits_total / expected_total
    oracle_pct= 100 * oracle_total / expected_total
    print(f"\n  Baseline (semantic only): {sem_pct:.1f}%")
    print(f"  Auto mode (current):      {auto_pct:.1f}%")
    print(f"  Oracle ceiling:           {oracle_pct:.1f}%")
    print(f"  Auto gap to oracle:       {oracle_pct - auto_pct:.1f}pp")

    print(f"\n  Total queries: {len(query_results)}")
    print(f"  Factual-routed queries: {mode_dist['factual']}")
    print(f"  Semantic-path queries:  {mode_dist['semantic']}")


if __name__ == "__main__":
    run()
