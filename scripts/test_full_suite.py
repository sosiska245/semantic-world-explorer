"""
Full 170-query test suite across brand / auto / semantic modes.
Run from project root: .venv/bin/python scripts/test_full_suite.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from src.data_loader import ENTITIES_DF, N_ENTITIES
from src.similarity import embed_query
from src.brand_sim import brand_sim
from src.multivec import apply_length_penalty, multi_sim
from src.bm25 import blend as bm25_blend
from src.domain_sim import domain_sim
from src.factual_router import detect_route, factual_scores

iso3  = ENTITIES_DF["iso3"].tolist()
names = ENTITIES_DF["name"].tolist()

HYBRID_TERMS = {"wheat","barley","cocoa","cacao","viticulture","paddy",
   "livestock","coffee production","wine production","tea production","fishing industry"}

def score_mode(mode, text, vec):
    if mode == "brand":
        return brand_sim(vec), "brand"
    if mode == "semantic":
        s = multi_sim(query_vec=vec, alpha=0.30)
        return apply_length_penalty(s), "semantic"
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

def top5(sc):
    return [names[i] for i in np.argsort(sc)[::-1][:5]]

def rank_of(sc, exp):
    if not exp: return None
    for r, i in enumerate(np.argsort(sc)[::-1]):
        if names[i].lower() == exp.lower(): return r + 1
    return -1

def conf(sc):
    s = sorted(sc, reverse=True)
    d = float(s[0] - s[19]) if len(s) >= 20 else 0.0
    lbl = "HIGH" if d > 0.10 else ("MED" if d > 0.05 else "LOW")
    return lbl, round(d, 4)

QUERIES = [
# ── BRAND (30) ──────────────────────────────────────────────────────────────
("anime manga pop culture",            "brand", "Japan"),
("K-pop Korean wave music",            "brand", "South Korea"),
("reggae music Bob Marley",            "brand", "Jamaica"),
("tango dance Buenos Aires",           "brand", "Argentina"),
("flamenco Spain Andalucia",           "brand", "Spain"),
("LEGO toy bricks",                    "brand", "Denmark"),
("IKEA furniture design",              "brand", "Sweden"),
("Ferrari Lamborghini sports cars",    "brand", "Italy"),
("Swiss chocolate luxury confectionery","brand", "Switzerland"),
("Bollywood Hindi film industry",      "brand", "India"),
("Nollywood African cinema",           "brand", "Nigeria"),
("Hollywood film movies entertainment","brand", "United States"),
("sushi ramen Japanese food",          "brand", "Japan"),
("espresso coffee cafe culture",       "brand", "Italy"),
("champagne sparkling wine bubbles",   "brand", "France"),
("Trappist craft beer Belgium",        "brand", "Belgium"),
("pilsner lager beer Bohemia",         "brand", "Czechia"),
("rose oil perfume attar Bulgaria",    "brand", "Bulgaria"),
("Formula One Grand Prix racing",      "brand", "United Kingdom"),
("rugby All Blacks haka",              "brand", "New Zealand"),
("carnival samba Rio festival",        "brand", "Brazil"),
("Muay Thai kickboxing martial art",   "brand", "Thailand"),
("eagle hunting nomadic Kyrgyzstan",   "brand", "Kyrgyzstan"),
("duduk woodwind Armenia music",       "brand", "Armenia"),
("kava ceremony Fiji Pacific",         "brand", "Fiji"),
("yurt nomadic steppe Mongolia",       "brand", "Mongolia"),
("Oktoberfest beer Bavaria Germany",   "brand", "Germany"),
("Fado music Portugal melancholy",     "brand", "Portugal"),
("capoeira Afro-Brazilian martial art","brand", "Brazil"),
("Diwali festival lights India",       "brand", "India"),
# ── TRAVEL (30) ─────────────────────────────────────────────────────────────
("fjords glaciers dramatic scenery",        "travel", "Norway"),
("tropical beach overwater bungalow",       "travel", "Maldives"),
("safari wildlife Big Five Africa",         "travel", "Kenya"),
("ancient ruins temples archaeology",       "travel", "Greece"),
("cherry blossom spring Japan",             "travel", "Japan"),
("Northern Lights aurora borealis",         "travel", "Norway"),
("street food night market Asia",           "travel", "Thailand"),
("ski resort Alpine mountains winter",      "travel", "Austria"),
("wine tasting vineyard cellar tour",       "travel", "France"),
("coral reef scuba diving tropical",        "travel", "Australia"),
("religious pilgrimage holy city Islam",    "travel", "Saudi Arabia"),
("backpacker budget travel Southeast Asia", "travel", "Thailand"),
("luxury safari lodge tented camp",        "travel", "Tanzania"),
("island hopping Caribbean beach",          "travel", "Bahamas"),
("ancient walled city medina souk",         "travel", "Morocco"),
("hot spring geothermal bathing pool",      "travel", "Iceland"),
("desert sand dunes Sahara camel",          "travel", "Morocco"),
("Buddhist temple monastery meditation",    "travel", "Thailand"),
("whale watching marine wildlife",          "travel", "Iceland"),
("volcano hiking active lava crater",       "travel", "Iceland"),
("tea plantation highland mist",            "travel", "Sri Lanka"),
("ancient spice market bazaar Istanbul",    "travel", "Turkey"),
("cave diving cenote underground",          "travel", "Mexico"),
("cycling infrastructure bike friendly",    "travel", "Netherlands"),
("waterfall jungle rainforest trekking",    "travel", "Costa Rica"),
("carnival costume parade festival",        "travel", "Brazil"),
("medieval castle fortress city",           "travel", "Czech Republic"),
("salt flat mirror reflection Uyuni",       "travel", "Bolivia"),
("river cruise historic bridges",           "travel", "Austria"),
("whale shark encounter swimming",          "travel", "Philippines"),
# ── CULTURE / FOOD / MUSIC / SPORTS (30) ────────────────────────────────────
("classical music opera symphony Vienna",   "culture", "Austria"),
("jazz blues soul American roots",          "culture", "United States"),
("hip hop rap urban culture",               "culture", "United States"),
("folk dance traditional costume",          "culture", "Hungary"),
("coffee origin specialty Ethiopia",        "culture", "Ethiopia"),
("spice cinnamon pepper Sri Lanka",         "culture", "Sri Lanka"),
("cocoa chocolate West Africa production",  "culture", "Ivory Coast"),
("tea ceremony ritual culture",             "culture", "Japan"),
("cheese wine French cuisine gourmet",      "culture", "France"),
("pasta pizza Mediterranean Italian",       "culture", "Italy"),
("kebab shawarma street food",              "culture", "Turkey"),
("tapas wine bar culture Spain",            "culture", "Spain"),
("cricket test match South Asia",           "culture", "India"),
("football soccer samba style champions",   "culture", "Brazil"),
("sumo wrestling Japan martial art",        "culture", "Japan"),
("chess grandmaster classical tournament",  "culture", "Russia"),
("cycling Tour de France peloton",          "culture", "France"),
("marathon long distance running East Africa","culture","Kenya"),
("carnival masquerade steelpan music",      "culture", "Trinidad and Tobago"),
("polyphonic choral singing tradition",     "culture", "Georgia"),
("throat singing khoomei overtone",         "culture", "Mongolia"),
("tango milonga dance halls",               "culture", "Argentina"),
("batik textile ikat fabric craft",         "culture", "Indonesia"),
("yoga ayurveda Vedic wellness India",      "culture", "India"),
("sake rice wine fermented Japan",          "culture", "Japan"),
("kimchi fermented vegetable Korea",        "culture", "South Korea"),
("ramen noodle soup broth Japan",           "culture", "Japan"),
("bagpipe highland music Celtic",           "culture", "United Kingdom"),
("mural fresco sacred art painting",        "culture", "Mexico"),
("polo horse sport equestrian",             "culture", "Argentina"),
# ── FACTUAL / RANKING (20) ──────────────────────────────────────────────────
("military spending defense budget",    "factual", "United States"),
("nuclear weapons warheads arsenal",    "factual", "Russia"),
("oil gas petroleum exporter",          "factual", "Saudi Arabia"),
("renewable energy solar wind capacity","factual", "China"),
("internet users connectivity",         "factual", "China"),
("life expectancy longevity health",    "factual", "Japan"),
("years schooling education level",     "factual", "Germany"),
("tourist arrivals visitors",           "factual", "France"),
("airline passengers aviation hub",     "factual", "United States"),
("health expenditure spending GDP",     "factual", "United States"),
("wheat barley grain export",           "factual", "Russia"),
("democracy freedom press index",       "factual", "Norway"),
("happiness wellbeing satisfaction",    "factual", "Finland"),
("carbon emissions climate footprint",  "factual", "China"),
("passport visa travel freedom",        "factual", "Japan"),
("startup unicorn tech investment",     "factual", "United States"),
("navy aircraft carrier warship fleet", "factual", "United States"),
("Olympic gold medals",                 "factual", "United States"),
("cocoa bean export production",        "factual", "Ivory Coast"),
("space exploration satellites launch", "factual", "United States"),
# ── SEMANTIC / GEOGRAPHY / POLITICS (20) ────────────────────────────────────
("archipelago island chain Pacific",        "semantic", "Indonesia"),
("Nordic welfare state social democracy",   "semantic", "Sweden"),
("constitutional monarchy parliament",      "semantic", "United Kingdom"),
("communist one party authoritarian",       "semantic", "China"),
("tax haven offshore banking secrecy",      "semantic", "Luxembourg"),
("manufacturing electronics export hub",    "semantic", "South Korea"),
("digital governance e-residency Skype",    "semantic", "Estonia"),
("rainforest deforestation Amazon basin",   "semantic", "Brazil"),
("canal waterway strategic chokepoint",     "semantic", "Panama"),
("ancient empire Nile civilization pyramid","semantic", "Egypt"),
("earthquake seismic ring of fire Pacific", "semantic", "Japan"),
("monsoon rice paddy flood delta",          "semantic", "Bangladesh"),
("refugee asylum humanitarian crisis",      "semantic", "Germany"),
("carbon neutral environmental biodiversity","semantic","Bhutan"),
("post-Soviet transition democracy",        "semantic", "Ukraine"),
("medieval gothic baroque historic centre", "semantic", "Czech Republic"),
("peninsula maritime port shipping",        "semantic", "Singapore"),
("theocracy Islamic republic clergy",       "semantic", "Iran"),
("federal multicultural immigration",       "semantic", "Canada"),
("microfinance rural poverty development",  "semantic", "Bangladesh"),
# ── EDGE CASES (20) ─────────────────────────────────────────────────────────
("xyzzy nonsense gibberish",                "edge", None),
("123 456 numbers only",                    "edge", None),
("the best country ever",                   "edge", None),
("penguins ice cold south pole",            "edge", None),
("country shaped like a boot",              "edge", "Italy"),
("smallest country world population",       "edge", "Vatican City"),
("where Santa Claus reindeer from",         "edge", "Finland"),
("pirates sea robbery ship hijack",         "edge", None),
("nuclear test explosion radiation legacy", "edge", None),
("world most remote island uninhabited",    "edge", "Bouvet Island"),
("country with most languages spoken",      "edge", "Papua New Guinea"),
("double landlocked by landlocked nations", "edge", "Liechtenstein"),
("red crabs migrate island annually",       "edge", "Christmas Island"),
("door to hell burning gas crater desert",  "edge", "Turkmenistan"),
("country entirely surrounded by Italy",    "edge", "Vatican City"),
("world largest salt flat mirror Bolivia",  "edge", "Bolivia"),
("coelacanth living fossil fish ocean",     "edge", "Comoros"),
("dodo extinction island Indian Ocean",     "edge", "Mauritius"),
("overwater bungalow paradise lagoon",      "edge", "Maldives"),
("Angel Falls highest waterfall world",     "edge", "Venezuela"),
]

print(f"Running {len(QUERIES)} queries …")
rows = []
for i, (query, cat, expected) in enumerate(QUERIES):
    vec = embed_query(query)
    row = {"q": query, "cat": cat, "exp": expected}
    for mode in ["auto", "brand", "semantic"]:
        sc, tag = score_mode(mode, query, vec)
        cl, cd  = conf(sc.tolist())
        t5      = top5(sc)
        rk      = rank_of(sc, expected)
        row[mode] = {"tag": tag, "top5": t5, "conf": cl, "delta": cd, "rank": rk}
    rows.append(row)
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(QUERIES)}")

print(f"Done: {len(rows)} rows")

def to_py(o):
    if isinstance(o, dict):  return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list):  return [to_py(x) for x in o]
    if isinstance(o, (np.floating,)):  return float(o)
    if isinstance(o, (np.integer,)):   return int(o)
    return o

rows = to_py(rows)
out = os.path.join(ROOT, "data", "processed", "test_suite_results.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Saved → {out}")
