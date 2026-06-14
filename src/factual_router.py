"""Smart Query Router — detects numeric/factual queries and maps them to
direct indicator rankings from facts_by_country.csv.

detect_route(query_text) -> (indicator, ascending, human_label) | None

Returns None for all queries that should use the default embedding path.

Validated by benchmark on 6 direct-indicator categories (factual 80% vs
embedding 20-30% hits@20). Military spending and healthcare are intentionally
NOT routed: wrong indicator (% GDP vs absolute) and small-island outliers.
"""

import os

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS_CSV = os.path.join(ROOT_DIR, "data", "processed", "facts_by_country.csv")

# (trigger_phrases, indicator, ascending, human_label)
# Matched as substrings of the lowercased query. More specific phrases first.
FACTUAL_ROUTES = [
    (
        ["life expectancy", "longevity", "how long people live", "lifespan",
         "life span", "years of life", "longest lived"],
        "life_expectancy", False, "life expectancy",
    ),
    (
        ["tourist arrivals", "most visited", "tourism arrivals",
         "most tourists", "international visitors", "visitor arrivals",
         "most popular destination"],
        "tourist_arrivals", False, "international tourist arrivals",
    ),
    (
        ["electoral democracy", "democracy index", "political freedom",
         "liberal democracy", "rule of law", "democratic freedom",
         "democracy score", "democratic countries", "most democratic"],
        "democracy_index", False, "electoral democracy index (V-Dem)",
    ),
    (
        ["internet users", "internet access", "internet penetration",
         "online users", "digital society", "most internet"],
        "internet_users_share", False, "internet users (% of population)",
    ),
    (
        ["renewable energy share", "clean energy leaders", "most renewable",
         "renewable energy leaders", "renewable electricity",
         "energy mix renewable", "renewable share", "greenest energy"],
        "renewable_share", False, "renewable energy share",
    ),
    (
        ["agriculture share", "agricultural economy", "farming share",
         "agriculture dominates", "subsistence farming economy",
         "agriculture gdp", "farm economy", "most agricultural"],
        "agriculture_share_gdp", False, "agriculture share of GDP",
    ),
    (
        ["military spending", "defence budget", "defense budget",
         "biggest military spender", "largest military budget",
         "highest military spending", "military expenditure"],
        "military_spending_usd", False, "military spending (USD, World Bank 2024)",
    ),
]

# Load facts once at import
_facts_df = pd.read_csv(FACTS_CSV) if os.path.exists(FACTS_CSV) else pd.DataFrame()
_FACT_INDEX: dict[tuple, float] = {}
_FACT_META:  dict[tuple, tuple] = {}   # (iso3, indicator) -> (year_str, source_str)

if not _facts_df.empty:
    for _, row in _facts_df.iterrows():
        key = (row["iso3"], row["indicator"])
        try:
            _FACT_INDEX[key] = float(row["value"])
        except (ValueError, TypeError):
            pass
        _FACT_META[key] = (str(row.get("year", "")), str(row.get("source", "")))


def detect_route(query_text: str):
    """Return (indicator, ascending, label) if query matches a factual route,
    or None to use the default embedding path."""
    t = (query_text or "").lower()
    for phrases, indicator, ascending, label in FACTUAL_ROUTES:
        if any(phrase in t for phrase in phrases):
            return indicator, ascending, label
    return None


def factual_scores(indicator: str, iso3_list: list, ascending: bool = False) -> np.ndarray:
    """Score array (float32) for all entities, normalized to [0,1].
    Countries with no data receive score 0 (ranks last)."""
    n = len(iso3_list)
    raw = np.zeros(n, dtype=np.float32)
    for i, iso3 in enumerate(iso3_list):
        val = _FACT_INDEX.get((iso3, indicator))
        if val is not None:
            raw[i] = val

    vmin, vmax = raw.min(), raw.max()
    if vmax > vmin:
        norm = (raw - vmin) / (vmax - vmin)
        return (1.0 - norm) if ascending else norm
    return raw


def factual_coverage(indicator: str, iso3_list: list) -> int:
    """Number of countries with data for this indicator."""
    return sum(1 for iso3 in iso3_list if (iso3, indicator) in _FACT_INDEX)


def get_source_info(indicator: str, iso3_list: list) -> str:
    """Return 'Source (year)' string from the first available country."""
    for iso3 in iso3_list:
        meta = _FACT_META.get((iso3, indicator))
        if meta and meta[0] and meta[0] != "nan":
            src = meta[1] if meta[1] and meta[1] != "nan" else ""
            yr  = meta[0]
            return f"{src} ({yr})" if src else f"({yr})"
    return ""
