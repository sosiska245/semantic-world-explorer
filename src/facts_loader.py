"""Load the precomputed structured-facts table for the detail panel's
"Evidence" block (data/processed/facts_by_country.csv, written by
scripts/fetch_facts.py and copied alongside embeddings by
scripts/build_embeddings.py).

This is a separate lookup table, not part of the embedded profile text -
it backs the "factual evidence" / "no structured data" distinction in the
UI, independent of semantic similarity.
"""

import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS_CSV = os.path.join(ROOT_DIR, "data", "processed", "facts_by_country.csv")


def _load_facts():
    if not os.path.exists(FACTS_CSV):
        return {}
    df = pd.read_csv(FACTS_CSV)
    facts = {}
    for iso3, group in df.groupby("iso3"):
        facts[iso3] = {
            cat: sub[["indicator", "value", "unit", "year", "source"]].to_dict("records")
            for cat, sub in group.groupby("category")
        }
    return facts


FACTS_BY_COUNTRY = _load_facts()


def get_facts(iso3, category):
    """List of {indicator, value, unit, year, source} dicts for this
    iso3+category, or None if no facts exist for that combination (the
    "missing data" case, to be distinguished from an empty list)."""
    return FACTS_BY_COUNTRY.get(iso3, {}).get(category)


INDICATOR_LABELS = {
    "renewable_share": "Renewable energy share",
    "agriculture_share_gdp": "Agriculture share of GDP",
    "agriculture_employment_share": "Agricultural employment share",
    "democracy_index": "Electoral democracy index",
    "regime_type": "Political regime",
    "population_density": "Population density",
    "land_area_km2": "Land area",
    "life_expectancy":        "Life expectancy",
    "health_expenditure_gdp": "Healthcare spending",
    "internet_users_share":   "Internet users",
    "mean_years_schooling":   "Mean years of schooling",
    "tourist_arrivals":       "International tourist arrivals",
    "air_passengers":         "Air transport passengers",
    "military_expenditure_gdp": "Military expenditure",
    "military_spending_usd":   "Military spending",
}

# value/unit -> display string per indicator (unit strings are formatted for
# direct display, e.g. "%" / "% of GDP" / "(0-1 scale)" / "people/km²").
INDICATOR_DISPLAY = {
    "renewable_share": lambda v, unit: f"{float(v):g}{unit}",
    "agriculture_share_gdp": lambda v, unit: f"{float(v):g}{unit}",
    "agriculture_employment_share": lambda v, unit: f"{float(v):g}{unit}",
    "democracy_index": lambda v, unit: f"{float(v):.2f} {unit}",
    "regime_type": lambda v, unit: str(v),
    "population_density": lambda v, unit: f"{float(v):g} {unit}",
    "land_area_km2": lambda v, unit: f"{float(v):,.0f} {unit}",
    "life_expectancy":        lambda v, unit: f"{float(v):g} {unit}",
    "health_expenditure_gdp": lambda v, unit: f"{float(v):g}{unit}",
    "internet_users_share":   lambda v, unit: f"{float(v):g}{unit}",
    "mean_years_schooling":   lambda v, unit: f"{float(v):g} {unit}",
    "tourist_arrivals":       lambda v, unit: f"{float(v):,.0f} {unit}",
    "air_passengers":         lambda v, unit: f"{float(v)/1e6:.1f}M {unit}",
    "military_expenditure_gdp": lambda v, unit: f"{float(v):.1f}{unit}",
    "military_spending_usd": lambda v, unit: (
        f"${float(v)/1e9:.1f}B" if float(v) >= 1e9 else f"${float(v)/1e6:.0f}M"
    ),
}

# Phase-1 categories only (energy/agriculture/politics/geography) - matches
# the categories actually present in facts_by_country.csv. A query that
# matches none of these gets no Evidence block (pure semantic mode).
QUERY_CATEGORY_KEYWORDS = {
    "energy": ["energy", "renewable", "solar", "wind power", "electricity", "fossil fuel", "nuclear power"],
    "agriculture": ["agricultur", "farm", "crop", "livestock", "harvest"],
    "politics": ["politic", "democra", "government", "election", "regime", "autocra"],
    "geography": ["geography", "population density", "land area", "square kilometer", "square mile", "terrain"],
    "health": [
        "health", "healthcare", "medicine", "hospital", "life expectancy",
        "mortality", "disease", "wellness", "medical",
    ],
    "technology": [
        "technology", "tech", "internet", "digital", "software", "startup",
        "innovation", "robotics", "programming",
    ],
    "education": [
        "education", "school", "university", "literacy", "academic",
        "research", "learning", "student",
    ],
    "tourism": [
        "tourism", "tourist", "travel", "resort", "destination",
        "backpacking", "sightseeing", "vacation",
    ],
    "aviation": [
        "aviation", "airport", "airline", "passenger", "cargo", "shipping",
        "maritime", "transport", "landlocked",
    ],
    "military": [
        "military", "armed forces", "army", "navy", "weapons", "arms",
        "peacekeeping", "war", "conflict", "defence", "defense",
    ],
}


def match_category(query_text):
    """First Phase-1 category whose keyword appears in `query_text`
    (case-insensitive), or None if no category matches."""
    text = (query_text or "").lower()
    for category, keywords in QUERY_CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return None
